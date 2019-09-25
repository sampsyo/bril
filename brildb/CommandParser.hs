module CommandParser where

import BrilTypes
import Data.Maybe (fromMaybe)
import Text.Parsec hiding (runP)
import Text.ParserCombinators.Parsec.Language

import qualified Text.Parsec.Token as T

tokenParser = T.makeTokenParser emptyDef

identifier = T.identifier tokenParser
integer = fromInteger <$> T.integer tokenParser
natural = fromInteger <$> T.natural tokenParser
parens = T.parens tokenParser
reserved = T.reserved tokenParser
whiteSpace = T.whiteSpace tokenParser

parseCommand :: String -> Either String Command
parseCommand = showLeft . parse command ""
  where
    showLeft (Left x) = Left $ show x
    showLeft (Right x) = Right x

command = whiteSpace *> (identifier >>= continueCmd) <* eof

continueCmd s = case s of
    "run" -> return Run
    "step" -> Step <$> optionOr 1 natural
    "restart" -> return Restart
    "print" -> Print <$> identifier
    "scope" -> return Scope
    "assign" -> Assign <$> identifier <*> brilValue
    "breakpoint" ->
        Breakpoint <$> lineLabel <*> optionOr (BoolConst True) boolExpr
    "list" -> return List
    c -> many anyChar *> return (UnknownCommand c)

lineLabel = Left <$> identifier <|> Right <$> natural

boolExpr = BoolConst <$> boolConst
    <|> BoolVar <$> identifier
    <|> parens boolOp

boolOp = (identifier >>=) $ \s -> case s of
    "eq" -> EqOp <$> intExpr <*> intExpr
    "lt" -> LtOp <$> intExpr <*> intExpr
    "gt" -> GtOp <$> intExpr <*> intExpr
    "le" -> LeOp <$> intExpr <*> intExpr
    "ge" -> GeOp <$> intExpr <*> intExpr
    "not" -> NotOp <$> boolExpr
    "and" -> AndOp <$> boolExpr <*> boolExpr
    "or" -> OrOp <$> boolExpr <*> boolExpr

intExpr = IntConst <$> integer
    <|> IntVar <$> identifier
    <|> parens intOp

intOp = (identifier >>=) $ \s -> case s of
    "add" -> AddOp <$> intExpr <*> intExpr
    "mul" -> MulOp <$> intExpr <*> intExpr
    "sub" -> SubOp <$> intExpr <*> intExpr
    "div" -> DivOp <$> intExpr <*> intExpr

brilValue = BrilInt <$> natural
    <|> BrilBool <$> boolConst

boolConst = string "true" *> return True
    <|> string "false" *> return False

optionOr x p = fromMaybe x <$> optionMaybe p
