{-# LANGUAGE ForeignFunctionInterface, TypeApplications, DeriveLift, FlexibleInstances, TemplateHaskell, BlockArguments #-}
module Lib where

import Language.Haskell.TH.Lib
import Language.Haskell.TH.Syntax
import Prelude hiding (sum)
import Foreign.C.String
import Foreign.C
import Foreign.Ptr (Ptr, FunPtr, nullPtr)
import Data.Maybe (fromJust)
import System.Mem
import Foreign.Marshal.Array
import Foreign.Marshal.Alloc
import Foreign.ForeignPtr
import System.IO.Unsafe (unsafePerformIO)
import Foreign.ForeignPtr.Unsafe
import Data.List (intercalate)
import Control.Monad.State.Lazy
import Control.Monad (when)

-- #include "tensorflow/compiler/xla/hashan/c_api.h"

--finalizer XLA_Remove as ^
--{#pointer *Buffer as BufferPrim foreign finalizer XLA_Remove as ^ newtype#}

data BufferPrim
foreign import ccall "XLA_CreateBuffer1D" xlaCreateBuffer :: Ptr CFloat -> CInt -> IO (Ptr BufferPrim)
foreign import ccall "XLA_CreateBuffer2D" xlaCreateBuffer2D :: Ptr CFloat -> CInt -> CInt -> IO (Ptr BufferPrim)
foreign import ccall "XLA_Print" xlaPrint :: Ptr BufferPrim -> IO ()
foreign import ccall "XLA_Create" xlaCreate :: Ptr CChar -> CInt -> Ptr (Ptr BufferPrim) -> IO (Ptr BufferPrim)
foreign import ccall "XLA_Init" xlaInit :: IO ()
foreign import ccall "XLA_Destructor" xlaRemove :: CInt -> FunPtr (Ptr BufferPrim -> IO ())

type Code a = TExpQ a
type Size2D = (Int, Int)
data Size = Dim2R Int Int | Dim2 Int Int | Dim1 Int | Dim0
  deriving (Eq, Show, Lift)
data Buffer = Device (ForeignPtr BufferPrim) Size | Const Float
type TreeRaw = Tree (Int, Size)
type Tensor = Buffer
type Layer = (Code ((Tensor, Tensor, Tensor) -> Tensor), Code ((Tensor, Tensor, Tensor) -> Tensor -> Tensor -> (Tensor, Tensor, Tensor)))

data BinOp
  = Add
  | Multiply
  | Divide
  | Subtract
  | Maximum
  | Minimum
  | Power
  | Dot
  | ReduceAdd
  | Gather
  | Compare CompOp
  deriving (Eq, Lift)

data UnaryOp
  = Sine
  | Cosine
  | Tanh
  | Abs
  | Negate
  | Exponential
  | Sign
  | Log
  | Transpose
  | ReduceArgMax
  | Reshape Size
  | Broadcast Size
  deriving (Eq, Lift)

data CompOp
  = LT
  | GT
  | LE
  | GE
  | EQ
  | NE
  deriving (Show, Lift, Eq)

data Tree a
  = Constant Float
  | Identity Int
  | Parameter a
  | BinEval BinOp (Tree a) (Tree a)
  | UnaryEval UnaryOp (Tree a)
  deriving (Lift)

instance Num Buffer where
  a + b = run (binEval Add (Parameter a) (Parameter b))
  a - b = run (binEval Subtract (Parameter a) (Parameter b))
  a * b = run (binEval Multiply (Parameter a) (Parameter b))
  negate a = run (unaryEval Negate (Parameter a))
  abs a = run (unaryEval Abs (Parameter a))
  signum a = run (unaryEval Sign (Parameter a))                           
  fromInteger = Const . fromIntegral -- Constant (fromIntegral a)
  {-# INLINE (*) #-}
  {-# INLINE negate #-}
  {-# INLINE (+) #-}

a `eq` b = run (binEval (Compare Lib.EQ) (Parameter a) (Parameter b))

a @@ b = run (binEval Dot (Parameter a) (Parameter b))
gather a b = run (binEval Gather (Parameter a) (Parameter b))
reshape a s = run (unaryEval (Reshape s) (Parameter a))

instance Fractional Buffer where
  a / b = run (binEval Divide (Parameter a) (Parameter b))
  fromRational a = undefined

instance Floating Buffer where
  pi = undefined
  exp a = run (unaryEval Exponential (Parameter a))
  log a = run (unaryEval Log (Parameter a))
  sin a = run (unaryEval Sine (Parameter a))
  cos a = run (unaryEval Cosine (Parameter a))
  tanh a = run (unaryEval Tanh (Parameter a))
  sinh = undefined
  cosh = undefined
  asinh = undefined
  acosh = undefined
  atanh = undefined

getSize :: Buffer -> Size
getSize (Device a b) = b

change :: Tree Buffer -> State Int (TreeRaw, [Buffer])
change (Parameter (Const f)) = return (Constant f, [])
change (Parameter a@(Device _ size)) = do
  st <- get
  put (st + 1)
  return (Parameter (st, size), [a])
change (BinEval binOp treeA treeB) = do
  (treeA', result) <- change treeA
  (treeB', result') <- change treeB
  return (BinEval binOp treeA' treeB', result ++ result')
change (UnaryEval unOp tree) = do 
  (tree', result) <- change tree
  return (UnaryEval unOp tree', result)
change (Constant f) = do
  return (Constant f, [])
change (Identity n) = do
  return (Identity n, [])

run :: Tree Buffer -> Buffer
run tree = unsafePerformIO do
  let ((tree', xs), _) = runState (change tree) 0
  let (code, size) = evalTop tree'
  putStrLn "running hlo"
  code' <- newCString code
  arr <- newArray (map (\(Device x _) -> unsafeForeignPtrToPtr x) xs)
  --putStrLn $ "len " ++ show (length xs)
  t <- xlaCreate code' (fromIntegral (length xs)) arr
  t' <- newForeignPtr (xlaRemove (fromIntegral 0)) t
  return $ Device t' size

evalTop :: TreeRaw -> (String, Size)
evalTop tree = (unlines $ [
  "HloModule xla_comp.0", 
  "",
  "primitive_computation_add.0 {",
  "  parameter.1 = f32[] parameter(0)",
  "  parameter.2 = f32[] parameter(1)",
  "  ROOT add.3 = f32[] add(parameter.1, parameter.2)",
  "}",
  "",
  "primitive_computation_argmax.0 {",
  "  parameter.1 = f32[] parameter(0)",
  "  parameter.2 = f32[] parameter(1)",
  "  parameter.3 = f32[] parameter(2)",
  "  parameter.4 = f32[] parameter(3)",
  "  compare.5 = pred[] compare(parameter.1, parameter.3), direction=GT",
  "  select.6 = f32[] select(compare.5, parameter.1, parameter.3)",
  "  select.7 = f32[] select(compare.5, parameter.2, parameter.4) ",
  "  ROOT tuple.8 = (f32[], f32[]) tuple(select.6, select.7)",
  "}",
  "",
  "ENTRY xla_comp.0 {"] ++
  init result ++
  ["  ROOT" ++ tail (last result), -- "  ROOT tuple." ++ (show st) ++ " = (f32[2,2]{1,0}) " ++ loc ++ "",
  "}"], size)
  where ((result, loc, size), _) = runState (eval tree) 1

unary op = case op of
  Sine -> "sine"
  Cosine -> "cosine"
  Tanh -> "tanh"
  Abs -> "abs"
  Negate -> "negate"
  Exponential -> "exponential"
  Sign -> "sign"
  Log -> "log"
  Transpose -> "transpose"
  Reshape _ -> "reshape"
  Broadcast _ -> "broadcast"

binary =
  [ (Add, "add"),
    (Multiply, "multiply"),
    (Divide, "divide"),
    (Subtract, "subtract"),
    (Maximum, "maximum"),
    (Minimum, "minimum"),
    (Power, "power")
  ]

len :: Buffer -> Int
len (Device _ (Dim1 m)) = m

rows :: Buffer -> Int
rows (Device _ (Dim2 m _)) = m
rows (Device _ (Dim2R m _)) = m

cols :: Buffer -> Int
cols (Device _ (Dim2 _ n)) = n
cols (Device _ (Dim2R _ n)) = n

showSize :: Size -> String
showSize = showSize' "f32"

showSize' :: String -> Size -> String
showSize' op (Dim2R a b) = op ++ "[" ++ show a ++ "," ++ show b ++ "]{0,1}"
showSize' op (Dim2 a b) = op ++ "[" ++ show a ++ "," ++ show b ++ "]{1,0}"
showSize' op (Dim1 a) = op ++ "[" ++ show a ++ "]{0}"
showSize' op Dim0 = op ++ "[]"

complete :: Size -> String -> [String] -> [String] -> [(String, String)] -> State Int ([String], String, Size)
complete size opStr prev args params = do
  st <- get
  put (st + 1)
  let loc = opStr ++ "." ++ show st
  let argStr = "(" ++ (intercalate ", " args) ++ ")"
  let paramStr = concatMap (\(k, v) -> ", " ++ k ++ "=" ++ v) params
  let cmd = "  " ++ loc ++ " = " ++ showSize size ++ " " ++ opStr ++ argStr ++ paramStr
  return (prev ++ [cmd], loc, size)


norm (Dim2R m1 m2) = Dim2 m1 m2
norm x = x

compat' Dim0 Dim0 = True
compat' (Dim1 m) (Dim1 n) = m == n
compat' (Dim2 m1 m2) (Dim2 n1 n2) = m1 == n1 && m2 == n2
compat' _ _ = False
--compat' m n = error $ "here " ++ show m ++ ", " ++ (show n)

compat m n = compat' (norm m) (norm n)

eval :: TreeRaw -> State Int ([String], String, Size)
eval (Parameter (i, size)) = 
  complete size "parameter" [] [show i] []

eval (UnaryEval (Broadcast size) tree) = do
  (result, loc, size') <- eval tree
  let args = compat size' size
  complete size "broadcast" result [loc] args
  where compat Dim0 (Dim2R _ _) = error "Size Mismatch"
        compat Dim0 _ = [("dimensions", "{}")]
        compat (Dim1 m) (Dim2 n1 n2)
          | m == n1 && m == n2 = error "Ambiguous"
          | m == n2 = [("dimensions", "{1}")]
          | m == n1 = [("dimensions", "{0}")]
        compat m n = error $ "Size Mismatch " ++ show m ++ ", " ++ show n

eval (UnaryEval Transpose tree) = do
  (result, loc, size) <- eval tree
  let size' = case size of
                Dim2 a b -> Dim2R b a
                Dim2R a b -> Dim2 b a
  complete size' "transpose" result [loc] [("dimensions", "{1,0}")]

eval (UnaryEval ReduceArgMax tree) = do
  (result, loc, size) <- eval tree
  let Dim2 rows cols = norm size
  st <- get
  let op1 = "  constant." ++ show st ++ " = f32[] constant(-inf)"
  let op2 = "  constant." ++ show (st + 1) ++ " = f32[] constant(0)"
  let op3 = "  iota." ++ show (st + 2) ++ " = f32[" ++ show rows ++ "," ++ show cols ++ "] iota(), iota_dimension=1"
  let op4 = "  reduce." ++ show (st + 3) ++ " = (f32[" ++ show rows ++ "]{0}, f32[" ++ show rows ++ "]{0}) reduce(" 
              ++ loc ++ ", iota." ++ show (st + 2) ++ ", constant." ++ show st ++ ", " ++ "constant." 
              ++ show (st + 1) ++ "), dimensions={1}, to_apply=primitive_computation_argmax.0"
  put (st + 4)
  complete (Dim1 rows) "get-tuple-element" (result ++ [op1, op2, op3, op4]) ["reduce." ++ show (st + 3)] [("index", "1")]

eval (UnaryEval op tree) = do
  (result, loc, size) <- eval tree
  let opStr = unary op
  let size' = case op of
                Reshape s -> s
                _ -> size
  complete size' opStr result [loc] []

eval (BinEval (Compare op) treeA treeB) = do
  (result, loc, size) <- eval treeA
  (result', loc', size') <- eval treeB
  if size `compat` size'
    then do
      st <- get
      let compare = "  compare." ++ show st ++ " = " ++ showSize' "pred" size ++ " compare(" ++ loc ++ ", " ++ loc' ++ "), direction=" ++ show op 
      put (st + 1)
      complete size "convert" (result ++ result' ++ [compare]) ["compare." ++ show st] []
    else error "Size Mismatch"

eval (BinEval Dot treeA treeB) = do
  (result, loc, size) <- eval treeA
  (result', loc', size') <- eval treeB
  let outputSize = compat size size'
  let args = case outputSize of
               Dim0 -> [("lhs_contracting_dims", "{0}"), ("rhs_contracting_dims", "{0}")]
               Dim2 _ _ -> [("lhs_contracting_dims", "{1}"), ("rhs_contracting_dims", "{0}")]
  complete outputSize "dot" (result ++ result') [loc, loc'] args
  where compat m n = compat' (norm m) (norm n)
        compat' (Dim2 m1 m2) (Dim2 n1 n2)
          | m2 == n1 = Dim2 m1 n2
        compat' (Dim1 m) (Dim1 n)
          | m == n = Dim0
        compat' m n = error $ "Size Mismatch, " ++ show m ++ ", " ++ show n

eval (BinEval Gather treeA treeB) = do
  (result, loc, size) <- eval treeA
  (result', loc', size') <- eval treeB
  let outputRows = case size' of
            Dim2 m 1 -> m
            _ -> error "Size Mismatch"
  let outputCols = case size of
            Dim2 m n -> n
            _ -> error "Size Mismatch"
  st <- get
  let convert = "  convert." ++ show st ++ " = s32[" ++ show outputRows ++ ",1]{1,0} convert(" ++ loc' ++ ")" 
  put (st + 1)
  complete (Dim2 outputRows outputCols) "gather" (result ++ result' ++ [convert]) [loc, "convert." ++ show st] 
    [("offset_dims", "{1}"), ("collapsed_slice_dims", "{0}"), ("start_index_map","{0}"),
      ("index_vector_dim", "1"), ("slice_sizes", "{1," ++ show outputCols ++ "}")]

eval (BinEval ReduceAdd treeA treeB) = do
  (result, loc, size) <- eval treeA
  (result', loc', size') <- eval treeB
  when (size' /= Dim0) (error "Size Mismatch")
  let outputSize = case norm size of
                     Dim1 m -> Dim0
                     Dim2 m n -> Dim1 m
                     _ -> error "Size Mismatch"
  let dims = case outputSize of
               Dim1 _ -> "{1}"
               Dim0 -> "{0}"
  complete outputSize "reduce" (result ++ result') [loc, loc'] [("dimensions", dims), ("to_apply", "primitive_computation_add.0")]

eval (BinEval op treeA treeB) = do
  (result, loc, size) <- eval treeA
  (result', loc', size') <- eval treeB
  let opStr = fromJust $ lookup op binary
  if size `compat` size'
    then complete size opStr (result ++ result') [loc, loc'] []
    else error $ "Size Mismatch " ++ opStr ++ " " ++ show size ++ " " ++ show size'

eval (Constant f) = do
  complete Dim0 "constant" [] [show f] []

eval (Identity n) = do
  st <- get
  let size = show n ++ "," ++ show n
  let op1 = "  iota." ++ show st ++ " = s32[" ++ size ++ "] iota(), iota_dimension=0"
  let op2 = "  iota." ++ show (st + 1) ++ " = s32[" ++ size ++ "] iota(), iota_dimension=1"
  let op3 = "  compare." ++ show (st + 2) ++ " = pred[" ++ size ++ 
              "]{1,0} compare(iota." ++ show st ++ ", iota." ++ show (st + 1) ++ "), direction=EQ"
  put (st + 3)
  complete (Dim2 n n) "convert" [op1, op2, op3] ["compare." ++ show (st + 2)] []

create2D :: Size2D -> [Float] -> Buffer
create2D (m, n) list = unsafePerformIO do
  when (m * n /= length list) (error "Size Mismatch")
  arr <- newArray (map CFloat list)
  t <- xlaCreateBuffer2D arr (fromIntegral m) (fromIntegral n)
  t' <- newForeignPtr (xlaRemove (fromIntegral 0)) t
  free arr
  return $ Device t' (Dim2 m n)

create1D :: [Float] -> Buffer
create1D list = unsafePerformIO do
  arr <- newArray (map CFloat list)
  let len = length list
  t <- xlaCreateBuffer arr (fromIntegral len)
  t' <- newForeignPtr (xlaRemove (fromIntegral 0)) t
  free arr
  return $ Device t' (Dim1 len)

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

mul :: [TExpQ Buffer] -> TExpQ (Buffer -> Buffer)
mul [] = [|| \a -> a ||]
mul (x:xs) = [|| \a -> $$x * $$(mul xs) a ||]

printView (Device a _) = xlaPrint (unsafeForeignPtrToPtr a)
initialise = xlaInit

c = Const . fromIntegral
c' = Const

compute layers
  = [|| \weights targets input -> 
      let activations = $$(forwardPass layers) weights input in
      let preds = softmax (last activations) in
      --let loss = negate (mean (sum (log preds * targets))) in
      let delta = (preds - targets) / (broadcast' (rows input, 10) (c (rows input))) in
      $$(backwardPass (reverse layers)) (reverse weights) (reverse activations) delta
    ||]

forwardPass :: [Layer] -> Code ([(Tensor, Tensor)] -> Tensor -> [Tensor])
forwardPass [] = [|| \[] x -> [x] ||]
forwardPass ((l1, l2):ls) 
  = [|| \((x1, x2):xs) y ->
      let tmp = $$l1 (x1, x2, y) in
      y : $$(forwardPass ls) xs tmp
    ||]

backwardPass :: [Layer] -> Code ([(Tensor, Tensor)] -> [Tensor] -> Tensor -> [(Tensor, Tensor)])
backwardPass [] = [|| \[] [_] _ -> []||]
backwardPass ((l1, l2):ls)
  = [|| \((x1, x2):xs) (i1:i2:is) delta ->
        let (a, b, c) = $$l2 (x1, x2, i2) i1 delta in
        (a, b) : $$(backwardPass ls) xs (i2:is) c
    ||]

layers = [linearTanh, linearTanh, linear]

transpose a = run (unaryEval Transpose (Parameter a))
reduceArgMax a = run (unaryEval ReduceArgMax (Parameter a))

linearTanh :: Layer
linearTanh = (
    [|| \(w, b, x) -> tanh (x @@ w + (broadcast' (rows x, cols w) b)) ||], 
    [|| \(w, b, x) a deltaA -> 
        let deltaZ = deltaA * (broadcast' (rows a, cols a) 1 - square a) in
        let deltaW = transpose x @@ deltaZ in
        let deltaB = (broadcast' (1, rows x) 1) @@ deltaZ in
        let deltaX = deltaZ @@ transpose w in
        (deltaW, reshape deltaB (Dim1 (cols deltaB)), deltaX) ||]
  )

identity n = run (Identity n)
square x = x * x
broadcast s =  run . unaryEval (Broadcast (Dim1 s)) . Parameter
broadcast' (m, n) =  run . unaryEval (Broadcast (Dim2 m n)) . Parameter

linear :: Layer
linear = (
    [|| \(w, b, x) -> x @@ w + (broadcast' (rows x, cols w) b) ||], 
    [|| \(w, b, x) _ deltaZ -> 
        let deltaW = transpose x @@ deltaZ in
        let deltaB = (broadcast' (1, rows x) 1) @@ deltaZ in
        let deltaX = deltaZ @@ transpose w in
        (deltaW, reshape deltaB (Dim1 (cols deltaB)), deltaX) ||]
  )

sum x = run (binEval ReduceAdd (Parameter x) (Parameter 0))
mean x = sum x / c (len x)
softmax x = exp x / (broadcast' (rows x, cols x) (sum (exp x)))