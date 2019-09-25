{-# LANGUAGE KindSignatures, Rank2Types, TypeFamilies #-}

module Lens where

import Control.Applicative
import Data.Function
import Data.Functor.Identity

import qualified Data.Map as Map
import qualified Data.Sequence as Seq

type Lens a a' b b' = forall f. Functor f =>
    (b -> f b') -> a -> f a'

type Lens' a b = Lens a a b b

type Traversal a a' b b' = forall f. Applicative f =>
    (b -> f b') -> a -> f a'

type Traversal' a b = Traversal a a b b

type Gettable a b = (b -> Const b b) -> a -> Const b a

type Iso a b = forall f p. (Expandable p, Functor f) =>
    p b (f b) -> p a (f a)

data Expander a b a' b' = Expander (a' -> a) (b -> b')

class Expandable (p :: * -> * -> *) where
    expand :: (a' -> a) -> (b -> b') -> p a b -> p a' b'

instance Expandable (->) where
    expand f b p = b . p . f

instance Expandable (Expander a b) where
    expand f' b' (Expander f b) = Expander (f . f') (b' . b)

class Indexed m where
    type Index m
    type Value m
    ix :: Index m -> Traversal' m (Value m)

instance Indexed [a] where
    type Index [a] = Int
    type Value [a] = a
    ix i f []     = pure []
    ix 0 f (x:xs) = (: xs) <$> f x
    ix i f (x:xs)
      | i < 0     = pure (x:xs)
      | otherwise = (x :) <$> ix (pred i) f xs

instance Ord k => Indexed (Map.Map k v) where
    type Index (Map.Map k v) = k
    type Value (Map.Map k v) = v
    ix k f m = case Map.lookup k m of
        Just v -> (\v' -> Map.insert k v' m) <$> f v
        Nothing -> pure m

instance Indexed (Seq.Seq a) where
    type Index (Seq.Seq a) = Int
    type Value (Seq.Seq a) = a
    ix i f s = case Seq.lookup i s of
        Just x -> (\x' -> Seq.update i x' s) <$> f x

lens :: (a -> b) -> (a -> b' -> a') -> Lens a a' b b'
lens v m f a = m a <$> f (v a)

iso :: (a -> b) -> (b -> a) -> Iso a b
iso f b = expand f (fmap b)

view :: Gettable a b -> a -> b
view l = getConst . l Const

over :: Traversal a a' b b' -> (b -> b') -> a -> a'
over l f = runIdentity . l (Identity . f)

set :: Traversal a a' b b' -> b' -> a -> a'
set l = over l . const

from :: Iso a b -> Iso b a
from i = expand (runIdentity . b) $ fmap f
  where
    Expander f b = i $ Expander id Identity

at :: Ord k => k -> Lens' (Map.Map k v) (Maybe v)
at = flip Map.alterF

_fst :: Lens (a, b) (a', b) a a'
_fst = lens fst (\(_, b) a' -> (a', b))

_snd :: Lens (a, b) (a, b') b b'
_snd = lens snd (\(a, _) b' -> (a, b'))

_head :: Traversal' [a] a
_head = ix 0

_left :: Traversal (Either a b) (Either a' b) a a'
_left f (Left a) = Left <$> f a
_left f (Right b) = pure $ Right b

_right :: Traversal (Either a b) (Either a b') b b'
_right f (Left a) = pure $ Left a
_right f (Right b) = Right <$> f b
