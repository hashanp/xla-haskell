{-# LANGUAGE ForeignFunctionInterface, TypeApplications, FlexibleInstances, TemplateHaskell #-}
module Lib where

import Language.Haskell.TH.Lib
import Prelude hiding (sin)
import Foreign.C.String
import Foreign.C
import Foreign.Ptr (Ptr, nullPtr)
import Data.Maybe (fromJust)
import System.Mem
import Foreign.Marshal.Array
import Foreign.Marshal.Alloc
import System.IO.Unsafe (unsafePerformIO)

#include "tensorflow/compiler/xla/hashan/c_api.h"

--finalizer XLA_Remove as ^
{#pointer *Buffer as BufferPrim foreign newtype#}

type Buffer = Ptr BufferPrim

data BinOp
  = Add
  | Multiply
  | Divide
  | Subtract
  | Maximum
  | Minimum
  | Power
  deriving (Eq)

data UnaryOp
  = Sine
  | Cosine
  | Tanh
  | Abs
  | Negate
  | Exponential
  deriving (Eq)

data CompOp
  = LT
  | GT
  | LE
  | GE
  | EQ

data Tree a
  = Parameter a
  | BinEval BinOp (Tree a) (Tree a)
  | UnaryEval UnaryOp (Tree a)
  | Broadcast Float

instance Num (Tree Buffer) where
  a + b = Parameter (run (binEval Add a b))
  a - b = Parameter (run (binEval Subtract a b))
  a * b = Parameter (run (binEval Multiply a b))
  negate a = Parameter (run (unaryEval Negate a))
  abs a = Parameter (run (unaryEval Abs a))
  signum a = undefined                           
  fromInteger a = Broadcast (fromIntegral a)
  {-# INLINE (*) #-}

change :: Tree Buffer -> Int -> (Tree Int, [Buffer], Int)
change (Parameter a) st = (Parameter st, [a], st + 1)
change (BinEval binOp treeA treeB) st = (BinEval binOp treeA' treeB', result ++ result', st'')
  where (treeA', result, st') = change treeA st
        (treeB', result', st'') = change treeB st'
change (UnaryEval unOp tree) st = (UnaryEval unOp tree', result, st')
  where (tree', result, st') = change tree st
change (Broadcast f) st = (Broadcast f, [], st)

run' :: Tree Buffer -> IO Buffer
run' tree = do
  let (tree', xs, _) = change tree 0
  code <- newCString $ evalTop tree'
  arr <- newArray xs
  putStrLn $ "len " ++ show (length xs)
  return $ {# call pure XLA_Create as ^ #} code (fromIntegral (length xs)) arr

run = unsafePerformIO . run'

evalTop :: Tree Int -> String
evalTop tree = unlines $ [
  "HloModule xla_comp.0", 
  "", 
  "ENTRY xla_comp.0 {"] ++
  init result ++
  ["  ROOT " ++ last result, -- "  ROOT tuple." ++ (show st) ++ " = (f32[2,2]{1,0}) " ++ loc ++ "",
  "}"]
  where (result, loc, st) = eval tree 1

unary =
  [ (Sine, "sine"),
    (Cosine, "cosine"),
    (Tanh, "tanh"),
    (Abs, "abs"),
    (Negate, "negate"),
    (Exponential, "exponential")
  ]

binary =
  [ (Add, "add"),
    (Multiply, "multiply"),
    (Divide, "divide"),
    (Subtract, "subtract"),
    (Maximum, "maximum"),
    (Minimum, "minimum"),
    (Power, "power")
  ]

{-eval :: Tree -> Int -> ([String], Int)
eval Parameter -}
eval :: Tree Int -> Int -> ([String], String, Int)
eval (Parameter i) st = (["  " ++ loc ++ " = f32[2,2]{1,0} parameter(" ++ show i ++ ")"], loc, st + 1)
  where loc = "parameter." ++ show st
eval (UnaryEval op tree) st = (result ++ ["  " ++ loc' ++ " = f32[2,2]{1,0} " ++ opStr ++ "(" ++ loc ++ ")"], loc', st' + 1)
  where opStr = fromJust $ lookup op unary
        (result, loc, st') = eval tree st
        loc' = opStr ++ "." ++ show st'
eval (BinEval op treeA treeB) st = (result ++ result' ++ ["  " ++ loc'' ++ " = f32[2,2]{1,0} " ++ opStr ++ "(" ++ loc ++ ", " ++ loc' ++ ")"], loc'', st'' + 1)
  where opStr = fromJust $ lookup op binary
        (result, loc, st') = eval treeA st
        (result', loc', st'') = eval treeB st'
        loc'' = opStr ++ "." ++ show st''
eval (Broadcast f) st = (["  " ++ loc ++ " = f32[] constant(" ++ show f ++ ")", "  " ++ loc' ++ " = f32[2,2]{1,0} broadcast(" ++ loc ++ "), dimensions={}"], loc', st + 2)
  where loc = "constant." ++ show st
        loc' = "broadcast." ++ show (st + 1)

create' :: (Int, Int) -> [Float] -> IO Buffer
create' (m, n) list = do
  arr <- newArray (map CFloat list)
  t <- {# call unsafe XLA_CreateBuffer2D as ^ #} arr (fromIntegral m) (fromIntegral n)
  free arr
  return t
  {-let arr = listArray ((0, 0), (m, n)) list in
  let (_, ptr) = toForeignPtr arr in
  {# call pure XLA_CreateBuffer2D as ^ #} (unsafeCoerce ptr) (fromIntegral m) (fromIntegral n) -}
create a b = unsafePerformIO (create' a b)

binEval = BinEval
unaryEval = UnaryEval

{-# INLINE[1] binEval #-}
{-# INLINE[1] unaryEval #-}
{-# INLINE CONLIKE [1] run #-}

{-# RULES "fusable/aux1" forall x y z.
      binEval x y (Parameter (run z)) = binEval x y z ; #-}
{-# RULES "fusable/aux2" forall x y z.
      binEval x (Parameter (run y)) z = binEval x y z ; #-}
{-# RULES "fusable/aux3" forall x y.
      unaryEval x (Parameter (run y)) = unaryEval x y ; #-}

power :: (Num a) => Int -> TExpQ (a -> a)
power 0 = [|| const 1 ||]
power n = [|| \a -> a * $$(power (n - 1)) a ||]

printView = {# call unsafe XLA_Print as ^ #}
initialise = {# call unsafe XLA_Init as ^ #}