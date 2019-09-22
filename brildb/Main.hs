module Main where

import BrilTypes
import Control.Monad (when)
import Control.Monad.IO.Class (liftIO)
import Control.Monad.Trans.State
import Data.Aeson
import Data.Maybe (fromJust, fromMaybe)
import Data.List (intercalate)
import Lens
import System.Environment (getArgs)
import System.IO

import qualified Data.ByteString.Lazy as BS
import qualified Data.Map.Lazy as Map
import qualified Data.Sequence as Seq

main :: IO ()
main = do
    args <- getArgs
    jsonString <- case args of
        [x] -> BS.readFile x
        _   -> error "usage: brildb <filename>"
    let program = case eitherDecode jsonString of
                  Right p -> p
                  Left e  -> error e
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
    executeCommand cmd
    debugLoop

executeCommand :: String -> StateT DebugState IO ()
executeCommand s = case words s of
    ["run"] -> run
    ["step"] -> step True
    ["restart"] -> modify (initialState . Program . program)
    ["print", x] -> printVar x
    ["scope"] -> printScope
    _ -> return ()

printVar :: String -> StateT DebugState IO ()
printVar x = checkTerminated $ do
    val <- getVariable x
    case val of
        Just v -> liftIO $ putStrLn $ x ++ " = " ++ show v
        Nothing -> liftIO $ putStrLn "not in scope"

printScope :: StateT DebugState IO ()
printScope = checkTerminated $ do
    assoc <- gets $ Map.toList . variables . head . callStack
    mapM_ (\(k, v) -> liftIO $ putStrLn $ k ++ " = " ++ show v) assoc


run :: StateT DebugState IO ()
run = checkTerminated $ do
    step False
    done <- gets terminated
    atBreakpoint <- gets ((Yes ==) . breakpoint . nextInstruction)
    if not done && atBreakpoint then
        liftIO $ putStrLn "breakpoint"
    else
        run

terminated :: DebugState -> Bool
terminated = null . callStack

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
