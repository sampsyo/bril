{-# LANGUAGE OverloadedStrings, TupleSections #-}

module BrilTypes where

import Data.Aeson
import Data.Char (toLower)
import Data.Foldable (toList)
import Data.List (intercalate)
import Data.Maybe (fromJust, fromMaybe, isJust)
import Lens

import qualified Data.Map.Lazy as Map
import qualified Data.Sequence as Seq

newtype Program = Program { functions :: Map.Map String Function }
    deriving Show

instance FromJSON Program where
    parseJSON = withObject "Program" $ \v ->
        Program . Map.fromList . map (\f -> (name f, f)) <$>
        v .: "functions"

data Function = Function {
    name :: String,
    body :: Seq.Seq Instruction,
    labels :: Map.Map String Int
}

instance Show Function where
    show (Function n b _) = n ++ " {\n" ++
        foldr (\i s ->
            if isJust (label i) then
                show i ++ "\n" ++ s
            else
                "  " ++ show i ++ "\n" ++ s
        ) "}" b

instance FromJSON Function where
    parseJSON = withObject "Function" $ \v -> do
        n <- v .: "name"
        insts <- v .: "instrs"
        let labelMap =
                Map.fromListWithKey (\l -> error ("repeated label: " ++ l)) $
                map (over _fst (fromJust . label)) $
                filter (isJust . label . fst) $
                zip (toList insts) [0 ..]
        return $ Function n insts labelMap

_body :: Lens' Function (Seq.Seq Instruction)
_body = lens body $ \f b -> f { body = b }

data Instruction = Instruction {
    breakCondition :: BoolExpr,
    label :: Maybe String,
    op :: Maybe String,
    dest :: Maybe String,
    typ :: Maybe Type,
    value :: Maybe BrilValue,
    args :: [String]
}

instance Show Instruction where
    show inst = show (classify inst) ++ showBP (breakCondition inst)
      where
        showBP (BoolConst False) = ""
        showBP (BoolConst True) = " (B)"
        showBP e = " (B: " ++ show e ++ ")"

instance FromJSON Instruction where
    parseJSON = withObject "Instruction" $ \v -> Instruction (BoolConst False)
        <$> v .:? "label"
        <*> v .:? "op"
        <*> v .:? "dest"
        <*> v .:? "type"
        <*> v .:? "value"
        <*> (fromMaybe [] <$> v .:? "args")

_breakCondition :: Lens' Instruction BoolExpr
_breakCondition = lens breakCondition $ \i c -> i { breakCondition = c }

data ClassifiedInst =
    Label String
  | Const String Type BrilValue
  | ValueOp String Type String [String]
  | EffectOp String [String]

instance Show ClassifiedInst where
    show (Label l) = l ++ ":"
    show (Const d t v) =
        d ++ ": " ++ show t ++ " = const " ++ show v ++ ";"
    show (ValueOp d t o as) =
        d ++ ": " ++ show t ++ " = " ++ intercalate " " (o:as) ++ ";"
    show (EffectOp o as) = 
        intercalate " " (o:as) ++ ";"

classify :: Instruction -> ClassifiedInst
classify (Instruction b l o d t v a) =
    if isJust l then
        Label $ fromJust l
    else if o == Just "const" then
        Const (fromJust d) (fromJust t) (fromJust v)
    else if isJust d then
        ValueOp (fromJust d) (fromJust t) (fromJust o) a
    else
        EffectOp (fromJust o) a
data BrilValue = BrilInt Int | BrilBool Bool

instance Show BrilValue where
    show (BrilInt x) = show x
    show (BrilBool x) = map toLower $ show x

instance FromJSON BrilValue where
    parseJSON v@(Number _) = BrilInt <$> parseJSON v
    parseJSON v@(Bool _) = BrilBool <$> parseJSON v
    parseJSON _ = fail "Value"

data Type = IntType | BoolType

instance Show Type where
    show IntType = "int"
    show BoolType = "bool"

instance FromJSON Type where
    parseJSON = withText "Type" $ \s -> case s of
        "int" -> return IntType
        "bool" -> return BoolType
        _ -> fail "Type"

data DebugState = DebugState {
    program :: Map.Map String Function,
    callStack :: [FunctionState]
}

_program :: Lens' DebugState (Map.Map String Function)
_program = lens program $ \d p -> d { program = p }

_callStack :: Lens' DebugState [FunctionState]
_callStack = lens callStack $ \d c -> d { callStack = c }

data FunctionState = FunctionState {
    functionName :: String,
    position :: Int,
    variables :: Map.Map String BrilValue
}

_variables :: Lens' FunctionState (Map.Map String BrilValue)
_variables = lens variables $ \f v -> f { variables = v }

_position :: Lens' FunctionState Int
_position = lens position $ \f p -> f { position = p }

data Command =
    Run
  | Step Int
  | Restart
  | Print String
  | Scope
  | Assign String BrilValue
  | Breakpoint (Either String Int) BoolExpr
  | List
  | UnknownCommand String

data BoolExpr =
    BoolVar String
  | BoolConst Bool
  | EqOp IntExpr IntExpr
  | LtOp IntExpr IntExpr
  | GtOp IntExpr IntExpr
  | LeOp IntExpr IntExpr
  | GeOp IntExpr IntExpr
  | NotOp BoolExpr
  | AndOp BoolExpr BoolExpr
  | OrOp BoolExpr BoolExpr

instance Show BoolExpr where
    showsPrec p e = case e of
        BoolVar v -> showString v
        BoolConst b -> showString $ map toLower $ show b
        EqOp e1 e2 -> showBinop "eq " e1 e2
        LtOp e1 e2 -> showBinop "lt " e1 e2
        GtOp e1 e2 -> showBinop "gt " e1 e2
        LeOp e1 e2 -> showBinop "le " e1 e2
        GeOp e1 e2 -> showBinop "ge " e1 e2
        NotOp e1 -> showUnop "not " e1
        AndOp e1 e2 -> showBinop "and " e1 e2
        OrOp e1 e2 -> showBinop "or " e1 e2
      where
        showBinop op e1 e2 = showParen (p > 10) $
            showString op . showsPrec 11 e1 .
            showString " " . showsPrec 11 e2
        showUnop op e1 = showParen (p > 10) $
            showString op . showsPrec 11 e1

data IntExpr =
    IntVar String
  | IntConst Int
  | AddOp IntExpr IntExpr
  | MulOp IntExpr IntExpr
  | SubOp IntExpr IntExpr
  | DivOp IntExpr IntExpr

instance Show IntExpr where
    showsPrec p e = case e of
        IntVar v -> showString v
        IntConst i -> showString (show i)
        AddOp e1 e2 -> showBinop "add " e1 e2
        SubOp e1 e2 -> showBinop "sub " e1 e2
        MulOp e1 e2 -> showBinop "mul " e1 e2
        DivOp e1 e2 -> showBinop "div " e1 e2
      where
        showBinop op e1 e2 = showParen (p > 10) $
            showString op . showsPrec 11 e1 .
            showString " " . showsPrec 11 e2
