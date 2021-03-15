{-# LANGUAGE ForeignFunctionInterface, TypeApplications, DeriveLift, FlexibleInstances, TemplateHaskell #-}
module Lib where

import Language.Haskell.TH.Lib
import Language.Haskell.TH.Syntax
import Prelude hiding (sin)
import Foreign.C.String
import Foreign.C
import Foreign.Ptr (Ptr, nullPtr)
import Data.Maybe (fromJust)
import System.Mem
import Foreign.Marshal.Array
import Foreign.Marshal.Alloc
import System.IO.Unsafe (unsafePerformIO)
import Data.List (intercalate)
import Control.Monad.State.Lazy

#include "tensorflow/compiler/xla/hashan/c_api.h"

--finalizer XLA_Remove as ^
{#pointer *Buffer as BufferPrim foreign newtype#}

type Size2D = (Int, Int)
type Buffer = (Ptr BufferPrim, Size2D)
type TreeRaw = Tree (Int, Size2D)

data BinOp
  = Add
  | Multiply
  | Divide
  | Subtract
  | Maximum
  | Minimum
  | Power
  deriving (Eq, Lift)

data UnaryOp
  = Sine
  | Cosine
  | Tanh
  | Abs
  | Negate
  | Exponential
  deriving (Eq, Lift)

data CompOp
  = LT
  | GT
  | LE
  | GE
  | EQ

data Tree a
  = Parameter a
  | BinEval BinOp (Tree a) (Tree a)
  | MatMul (Tree a) (Tree a)
  | UnaryEval UnaryOp (Tree a)
  | Broadcast Float
  deriving (Lift)

instance Num (Tree Buffer) where
  a + b = Parameter (run (binEval Add a b))
  a - b = Parameter (run (binEval Subtract a b))
  a * b = Parameter (run (binEval Multiply a b))
  negate a = Parameter (run (unaryEval Negate a))
  abs a = Parameter (run (unaryEval Abs a))
  signum a = undefined                           
  fromInteger a = Broadcast (fromIntegral a)
  {-# INLINE (*) #-}
  {-# INLINE negate #-}
  {-# INLINE (+) #-}

getSize :: Buffer -> Size2D
getSize (a, b) = b

change :: Tree Buffer -> State Int (TreeRaw, [Buffer])
change (Parameter a) = do
  st <- get
  put (st + 1)
  return (Parameter (st, getSize a), [a])
change (BinEval binOp treeA treeB) = do
  (treeA', result) <- change treeA
  (treeB', result') <- change treeB
  return (BinEval binOp treeA' treeB', result ++ result')
change (UnaryEval unOp tree) = do 
  (tree', result) <- change tree
  return (UnaryEval unOp tree', result)
change (MatMul treeA treeB) = do
  (treeA', result) <- change treeA
  (treeB', result') <- change treeB
  return (MatMul treeA' treeB', result ++ result')
change (Broadcast f) = do 
  return (Broadcast f, [])

run' :: Tree Buffer -> IO Buffer
run' tree = do
  let ((tree', xs), _) = runState (change tree) 0
  let (code, size) = evalTop tree'
  code' <- newCString code
  arr <- newArray (map fst xs)
  putStrLn $ "len " ++ show (length xs)
  return $ ({# call pure XLA_Create as ^ #} code' (fromIntegral (length xs)) arr, size)

run = unsafePerformIO . run'

evalTop :: TreeRaw -> (String, Size2D)
evalTop tree = (unlines $ [
  "HloModule xla_comp.0", 
  "", 
  "ENTRY xla_comp.0 {"] ++
  init result ++
  ["  ROOT " ++ last result, -- "  ROOT tuple." ++ (show st) ++ " = (f32[2,2]{1,0}) " ++ loc ++ "",
  "}"], size)
  where ((result, loc, size), _) = runState (eval tree) 1

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

showSize :: Size2D -> String
showSize (a, b) = "f32[" ++ show a ++ "," ++ show b ++ "]{1,0}"

complete :: Size2D -> String -> [String] -> [String] -> [(String, String)] -> State Int ([String], String, Size2D)
complete size opStr prev args params = do
  st <- get
  put (st + 1)
  let loc = opStr ++ "." ++ show st
  let argStr = "(" ++ (intercalate ", " args) ++ ")"
  let paramStr = concatMap (\(k, v) -> ", " ++ k ++ "=" ++ v) params
  let cmd = "  " ++ loc ++ " = " ++ showSize size ++ " " ++ opStr ++ argStr ++ paramStr
  return (prev ++ [cmd], loc, size)

eval :: TreeRaw -> State Int ([String], String, Size2D)
eval (Parameter (i, size)) = 
  complete size "parameter" [] [show i] []

eval (UnaryEval op tree) = do
  (result, loc, size) <- eval tree
  let opStr = fromJust $ lookup op unary
  complete size opStr result [loc] []

eval (BinEval op treeA treeB) = do
  (result, loc, size) <- eval treeA
  (result', loc', size') <- eval treeB
  let opStr = fromJust $ lookup op binary
  if size /= size'
    then error "Size Mismatch"
    else complete size opStr (result ++ result') [loc, loc'] []

eval (MatMul treeA treeB) = do
  (result, loc, size) <- eval treeA
  (result', loc', size') <- eval treeB
  if size /= size'
    then error "Size Mismatch"
    else complete size "dot" (result ++ result') [loc, loc'] [("lhs_contracting_dims", "{1}"), ("rhs_contracting_dims", "{0}")]

eval (Broadcast f) = do
  st <- get
  let loc = "constant." ++ show st
  let loc' = "broadcast." ++ show (st + 1)
  put (st + 2)
  return (["  " ++ loc ++ " = f32[] constant(" ++ show f ++ ")", "  " ++ loc' ++ " = f32[2,2]{1,0} broadcast(" ++ loc ++ "), dimensions={}"], loc', (2, 2))

create' :: Size2D -> [Float] -> IO Buffer
create' (m, n) list = do
  arr <- newArray (map CFloat list)
  t <- {# call unsafe XLA_CreateBuffer2D as ^ #} arr (fromIntegral m) (fromIntegral n)
  free arr
  return (t, (m, n))
create a b = unsafePerformIO (create' a b)

{-let arr = listArray ((0, 0), (m, n)) list in
let (_, ptr) = toForeignPtr arr in
{# call pure XLA_CreateBuffer2D as ^ #} (unsafeCoerce ptr) (fromIntegral m) (fromIntegral n) -}

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

power ::  (Num a) => Int -> TExpQ (a -> a)
power 0 = [|| const 1 ||]
power n = [|| \a -> a * $$(power (n - 1)) a ||]

mul :: [TExpQ (Tree Buffer)] -> TExpQ (Tree Buffer -> Tree Buffer)
mul [] = [|| \a -> a ||]
mul (x:xs) = [|| \a -> $$x * $$(mul xs) a ||]

printView = {# call unsafe XLA_Print as ^ #} . fst
initialise = {# call unsafe XLA_Init as ^ #}