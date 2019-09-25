module Main where

import BrilTypes
import CommandParser
import Control.Monad (when)
import Control.Monad.IO.Class (liftIO)
import Control.Monad.Trans.State
import Data.Aeson
import Data.Maybe (fromJust, fromMaybe, isJust)
import Data.List (intercalate)
import Lens
import System.Environment (getArgs)
import System.IO
import Text.Read (readMaybe)

import qualified Data.ByteString.Lazy as BS
import qualified Data.Map.Lazy as Map
import qualified Data.Sequence as Seq

main :: IO ()
main = do
    args <- getArgs
    jsonString <- case args of
        [x] -> BS.readFile x
        _   -> errorWithoutStackTrace "usage: brildb <filename>"
    let program = case eitherDecode jsonString of
                  Right p -> p
                  Left e  -> errorWithoutStackTrace "invalid Bril file"
    hSetBuffering stdout NoBuffering
    evalStateT debugLoop $ initialState program

initialState :: Program -> DebugState
initialState p = DebugState (functions p) [initialFuncState "main"]

initialFuncState :: String -> FunctionState
initialFuncState f = FunctionState f 0 Map.empty

debugLoop :: StateT DebugState IO ()
debugLoop = do
    liftIO $ putStr "(brildb) "
    cmd <- liftIO getLine
    when (not $ null cmd) $ case parseCommand cmd of
        Right c -> executeCommand c
        Left err -> liftIO $ putStrLn err
    debugLoop

executeCommand :: Command -> StateT DebugState IO ()
executeCommand c = case c of
    Run -> run Nothing
    Step 1 -> step True
    Step n -> run $ Just n
    Restart -> modify $ initialState . Program . program
    Print x -> printVar x
    Scope -> printScope
    Assign x v -> checkTerminated $ setVariable x v
    Breakpoint p e -> setBreakpoint p e
    List -> listCmd
    UnknownCommand c -> liftIO $ putStrLn $ "unknown command: " ++ c

setBreakpoint :: Either String Int -> BoolExpr -> StateT DebugState IO ()
setBreakpoint (Left l) e = do
    pos <- gets $ Map.lookup l . labels . (Map.! "main") . program
    case pos of
        Nothing -> liftIO $ putStrLn $ "unknown label: " ++ l
        Just i -> setBreakpoint (Right i) e
setBreakpoint (Right i) e = modify $
    set (_program . ix "main" . _body . ix (pred i) . _breakCondition) e

printVar :: String -> StateT DebugState IO ()
printVar x = checkTerminated $ do
    val <- getVariable x
    case val of
        Just v -> liftIO $ putStrLn $ x ++ " = " ++ show v
        Nothing -> liftIO $ putStrLn $ "not in scope: " ++ x

printScope :: StateT DebugState IO ()
printScope = checkTerminated $ do
    assoc <- gets $ Map.toList . variables . head . callStack
    mapM_ (\(k, v) -> liftIO $ putStrLn $ k ++ " = " ++ show v) assoc

listCmd :: StateT DebugState IO ()
listCmd = checkTerminated $ do
    Function n b _ <- gets currentFunction
    pos <- gets $ position . head . callStack
    let numLength = length $ show $ Seq.length b
    let aux n i =
            (if pos == n then "-> " else "   ") ++
            rightAlign numLength (show $ succ n) ++
            (if isJust (label i) then "  " else "      ") ++
            show i ++ "\n"
    let funcString = concat [n, " {\n", Seq.foldMapWithIndex aux b, "}"]
    liftIO $ putStrLn funcString
  where
    rightAlign len s = replicate (len - length s) ' ' ++ s

-- requires n non-negative
run :: Maybe Int -> StateT DebugState IO ()
run (Just 0) = return ()
run n = checkTerminated $ do
    step False
    done <- gets terminated
    break <- atBreakpoint
    st <- get
    if not done && break then
        liftIO $ print $ nextInstruction st
    else
        run $ fmap pred n

terminated :: DebugState -> Bool
terminated = null . callStack

atBreakpoint :: StateT DebugState IO Bool
atBreakpoint = do
    cond <- gets $ breakCondition . nextInstruction
    vars <- gets $ variables . head . callStack
    case evalBool vars cond of
        Left s -> liftIO (putStrLn s) >> return True
        Right b -> return b

evalBool :: Map.Map String BrilValue -> BoolExpr -> Either String Bool
evalBool m e = case e of
    BoolVar x -> case Map.lookup x m of
        Nothing -> Left $ "unbound variable: " ++ x
        Just (BrilInt _) -> Left $ "expected bool but got int: " ++ x
        Just (BrilBool b) -> Right b
    BoolConst b -> Right b
    EqOp e1 e2 -> (==) <$> evalInt m e1 <*> evalInt m e2
    LtOp e1 e2 -> (<) <$> evalInt m e1 <*> evalInt m e2
    GtOp e1 e2 -> (>) <$> evalInt m e1 <*> evalInt m e2
    LeOp e1 e2 -> (<=) <$> evalInt m e1 <*> evalInt m e2
    GeOp e1 e2 -> (>=) <$> evalInt m e1 <*> evalInt m e2
    NotOp e -> not <$> evalBool m e
    AndOp e1 e2 -> (&&) <$> evalBool m e1 <*> evalBool m e2
    OrOp e1 e2 -> (||) <$> evalBool m e1 <*> evalBool m e2

evalInt :: Map.Map String BrilValue -> IntExpr -> Either String Int
evalInt m e = case e of
    IntVar x -> case Map.lookup x m of
        Nothing -> Left $ "unbound variable: " ++ x
        Just (BrilBool _) -> Left $ "expected int but got bool: " ++ x
        Just (BrilInt i) -> Right i
    IntConst i -> Right i
    AddOp e1 e2 -> (+) <$> evalInt m e1 <*> evalInt m e2
    MulOp e1 e2 -> (*) <$> evalInt m e1 <*> evalInt m e2
    SubOp e1 e2 -> (-) <$> evalInt m e1 <*> evalInt m e2
    DivOp e1 e2 -> div <$> evalInt m e1 <*> evalInt m e2

checkTerminated :: StateT DebugState IO () -> StateT DebugState IO ()
checkTerminated st = do
    term <- gets terminated
    if term then
        liftIO $ putStrLn "program terminated."
    else
        st

-- requires `not (terminated st)`
nextInstruction :: DebugState -> Instruction
nextInstruction st = case Map.lookup funcName prog of
    Just f -> Seq.index (body f) (position funcSt)
    Nothing -> error $ "call of unknown function: " ++ funcName
  where
    funcSt = head $ callStack st
    funcName = functionName funcSt
    prog = program st

step :: Bool -> StateT DebugState IO ()
step printInst = checkTerminated $ do
    inst <- gets nextInstruction
    when printInst $ liftIO $ print inst
    operation (fromMaybe "label" $ op inst) inst

operation :: String -> Instruction -> StateT DebugState IO ()
operation "const" = constOp
operation "add" = numericalBinop (+)
operation "mul" = numericalBinop (*)
operation "sub" = numericalBinop (-)
operation "div" = numericalBinop div
operation "eq" = numericalComp (==)
operation "lt" = numericalComp (<)
operation "gt" = numericalComp (>)
operation "le" = numericalComp (<=)
operation "ge" = numericalComp (>=)
operation "not" = booleanUnop not
operation "and" = booleanBinop (&&)
operation "or" = booleanBinop (||)
operation "jmp" = jumpOp
operation "br" = branchOp
operation "ret" = const returnOp
operation "id" = unop id
operation "print" = printOp
operation "nop" = const incrementPosition
operation "label" = const incrementPosition

constOp :: Instruction -> StateT DebugState IO ()
constOp inst = do
    setVariable (fromJust $ dest inst) (fromJust $ value inst) 
    incrementPosition

binop :: (BrilValue -> BrilValue -> BrilValue) ->
         Instruction -> StateT DebugState IO ()
binop f inst = do
    [x, y] <- mapM getVariable' $ assertNumArgs 2 $ args inst
    setVariable (fromJust $ dest inst) $ f x y
    incrementPosition

unop :: (BrilValue -> BrilValue) -> Instruction -> StateT DebugState IO ()
unop f inst = do
    [x] <- mapM getVariable' $ assertNumArgs 1 $ args inst
    setVariable (fromJust $ dest inst) $ f x
    incrementPosition

numericalBinop :: (Int -> Int -> Int) -> Instruction -> StateT DebugState IO ()
numericalBinop f = binop $ \x y -> BrilInt $ f (assertInt x) (assertInt y)

numericalComp :: (Int -> Int -> Bool) -> Instruction -> StateT DebugState IO ()
numericalComp f = binop $ \x y -> BrilBool $ f (assertInt x) (assertInt y)

booleanUnop :: (Bool -> Bool) -> Instruction -> StateT DebugState IO ()
booleanUnop f = unop $ \x -> BrilBool $ f $ assertBool x

booleanBinop :: (Bool -> Bool -> Bool) -> Instruction -> StateT DebugState IO ()
booleanBinop f = binop $ \x y -> BrilBool $ f (assertBool x) (assertBool y)

gotoLabel :: String -> StateT DebugState IO ()
gotoLabel label = do
    func <- gets currentFunction
    let pos = labels func Map.! label
    modify $ set (_callStack . _head . _position) pos

jumpOp :: Instruction -> StateT DebugState IO ()
jumpOp inst = gotoLabel (head $ assertNumArgs 1 $ args inst)  

branchOp :: Instruction -> StateT DebugState IO ()
branchOp inst = do
    let [var, lblTrue, lblFalse] = args inst
    b <- assertBool <$> getVariable' var
    if b then
        gotoLabel lblTrue
    else
        gotoLabel lblFalse

returnOp :: StateT DebugState IO ()
returnOp = do
    modify (over _callStack tail)

printOp :: Instruction -> StateT DebugState IO ()
printOp inst = do
    xs <- mapM getVariable' $ args inst
    liftIO $ putStrLn $ intercalate " " $ map show xs
    incrementPosition

assertNumArgs :: Int -> [a] -> [a]
assertNumArgs n xs
    | length xs == n = xs
    | otherwise = error $ "expected " ++ show n ++ " arguments"

assertInt :: BrilValue -> Int
assertInt (BrilInt x) = x
assertInt _ = error "expected int"

assertBool :: BrilValue -> Bool
assertBool (BrilBool x) = x
assertBool _ = error "expected bool"

setVariable :: String -> BrilValue -> StateT DebugState IO ()
setVariable x v = modify $
    over (_callStack . _head . _variables) $ Map.insert x v

getVariable :: String -> StateT DebugState IO (Maybe BrilValue)
getVariable x = gets $ Map.lookup x . variables . head . callStack

getVariable' :: String -> StateT DebugState IO BrilValue
getVariable' x = gets $ (Map.! x) . variables . head . callStack

getIntVar :: String -> StateT DebugState IO Int
getIntVar x = do
    BrilInt i <- getVariable' x
    return i

incrementPosition :: StateT DebugState IO ()
incrementPosition = do
    pos <- gets $ position . head . callStack
    numInsts <- gets $ Seq.length . body . currentFunction
    if succ pos == numInsts then
        returnOp
    else
        modify $ over (_callStack . _head . _position) succ

currentFunction :: DebugState -> Function
currentFunction (DebugState p (f:_)) = p Map.! functionName f
