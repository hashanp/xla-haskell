{-# LANGUAGE ForeignFunctionInterface, LambdaCase, TypeFamilies, UndecidableInstances, AllowAmbiguousTypes, TypeFamilyDependencies, DeriveFunctor, ScopedTypeVariables, GADTs, StandaloneDeriving, StandaloneKindSignatures, DataKinds, BangPatterns, TypeApplications, DeriveLift, FlexibleInstances, TemplateHaskell, BlockArguments #-}
module Lib where

-- AllowAmbiguousTypes

import Language.Haskell.TH.Lib
import Language.Haskell.TH.Syntax hiding (Type)
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
import Data.Kind (Type)
import Debug.Trace (trace)

-- #include "tensorflow/compiler/xla/hashan/c_api.h"

--finalizer XLA_Remove as ^
--{#pointer *Buffer as BufferPrim foreign finalizer XLA_Remove as ^ newtype#}

data BufferPrim
foreign import ccall "XLA_CreateBuffer1D" xlaCreateBuffer :: Ptr CFloat -> CInt -> IO (Ptr BufferPrim)
foreign import ccall "XLA_CreateBuffer2D" xlaCreateBuffer2D :: Ptr CFloat -> CInt -> CInt -> IO (Ptr BufferPrim)
foreign import ccall "XLA_Print" xlaPrint :: Ptr BufferPrim -> IO ()
foreign import ccall "XLA_Create" xlaCreate :: Ptr CChar -> CInt -> Ptr (Ptr BufferPrim) -> IO (Ptr (Ptr BufferPrim))
foreign import ccall "XLA_Init" xlaInit :: IO ()
foreign import ccall "XLA_Destructor" xlaRemove :: CInt -> FunPtr (Ptr BufferPrim -> IO ())

type Code a = TExpQ a
type Size2D = (Int, Int)
data SizePrim a = Dim2R a a | Dim2 a a | Dim1 a | Dim0
  deriving (Eq, Show, Lift, Functor)
type Size = SizePrim Int
type SizeExtra = SizePrim TreeSize
data Buffer = Device !(ForeignPtr BufferPrim) Size | Const Float
type TreeRaw = Tree (Int, Size)
type Tensor = Buffer
type Layer = (Code ((Tensor, Tensor, Tensor) -> Tensor), Code ((Tensor, Tensor, Tensor) -> Tensor -> Tensor -> (Tensor, Tensor, Tensor)))

thd (a, b, c) = c

makeIntoTreeSize :: Size -> SizeExtra
makeIntoTreeSize = fmap Real
extractFromTreeSize :: [Result] -> SizeExtra -> Size
extractFromTreeSize res = fmap (\case 
                              Real a -> a
                              FakeRows a -> rows' (thd (res !! (length res - a)))
                              FakeCols a -> cols' (thd (res !! (length res - a))))

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
  deriving (Eq, Lift, Show)

data TreeSize 
  = Real Int
  | FakeRows Int
  | FakeCols Int
  deriving (Eq, Lift, Show)

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
  | Reshape SizeExtra
  | Broadcast SizeExtra
  deriving (Eq, Lift, Show)

data CompOp
  = LT
  | GT
  | LE
  | GE
  | EQ
  | NE
  deriving (Show, Lift, Eq)

data TreePrim a
  = Constant Float
  | Identity Int
  -- | Parameter a
  | BinEval BinOp !a !a
  | UnaryEval UnaryOp !a
  | CrossRef Int
  deriving (Functor, Lift, Show)

data Free f a
  = Pure a
  | Impure (f (Free f a))
  deriving (Functor)

type Tree a = Free TreePrim a

deriving instance Show (TreeRaw)
deriving instance Show (Tree Buffer)

instance Show Buffer where
  show _ = "Buffer"
  --show (Device _ _) = "BufferDevice"
  --show (Const _) = "BufferConst"

instance Num Buffer where
  a + b = run (binEval Add (Pure a) (Pure b))
  a - b = run (binEval Subtract (Pure a) (Pure b))
  a * b = run (binEval Multiply (Pure a) (Pure b))
  negate a = run (unaryEval Negate (Pure a))
  abs a = run (unaryEval Abs (Pure a))
  signum a = run (unaryEval Sign (Pure a))                           
  fromInteger = Const . fromIntegral -- Constant (fromIntegral a)
  {-# INLINE (*) #-}
  {-# INLINE (-) #-}
  {-# INLINE negate #-}
  {-# INLINE (+) #-}

a `eq` b = run (binEval (Compare Lib.EQ) (Pure a) (Pure b))
a @@ b = run (binEval Dot (Pure a) (Pure b))
gather a b = run (binEval Gather (Pure a) (Pure b))
reshape a s = run (unaryEval (Reshape (makeIntoTreeSize s)) (Pure a))

instance Fractional Buffer where
  a / b = run (binEval Divide (Pure a) (Pure b))
  fromRational a = undefined
  {-# INLINE (/) #-}

instance Floating Buffer where
  pi = undefined
  exp a = run (unaryEval Exponential (Pure a))
  log a = run (unaryEval Log (Pure a))
  sin a = run (unaryEval Sine (Pure a))
  cos a = run (unaryEval Cosine (Pure a))
  tanh a = run (unaryEval Tanh (Pure a))
  sinh = undefined
  cosh = undefined
  asinh = undefined
  acosh = undefined
  atanh = undefined
  {-# INLINE exp #-}
  {-# INLINE tanh #-}
  {-# INLINE log #-}

getSize :: Buffer -> Size
getSize (Device a b) = b

change :: Tree Buffer -> State Int (TreeRaw, [Buffer])
change (Pure (Const f)) = return (Impure (Constant f), [])
change (Pure a@(Device _ size)) = do
  st <- get
  put (st + 1)
  return (Pure (st, size), [a])
change (Impure (BinEval binOp treeA treeB)) = do
  (treeA', result) <- change treeA
  (treeB', result') <- change treeB
  return (binEval binOp treeA' treeB', result ++ result')
change (Impure (UnaryEval unOp tree)) = do 
  (tree', result) <- change tree
  return (unaryEval unOp tree', result)
change (Impure (Constant f)) = do
  return (Impure (Constant f), [])
change (Impure (Identity n)) = do
  return (Impure (Identity n), [])
change (Impure (CrossRef n)) = do
  return (Impure (CrossRef n), [])

runHelper :: [Tree Buffer] -> State Int ([TreeRaw], [Buffer])
runHelper [] = return ([], [])
runHelper (x:xs) = do
  (tree, params) <- change x
  (tree', params') <- runHelper xs
  return (tree:tree', params ++ params')

runInner :: [Tree Buffer] -> [Buffer]
runInner trees = unsafePerformIO do
  let ((tree', xs), _) = runState (runHelper trees) 0
  let (code, size) = evalTop tree'
  --putStrLn "running hlo"
  code' <- newCString code
  arr <- newArray (map (\(Device x _) -> unsafeForeignPtrToPtr x) xs)
  -- putStrLn $ "len " ++ show (length xs)
  t <- xlaCreate code' (fromIntegral (length xs)) arr
  t' <- peekArray (length trees) t
  t'' <- mapM (newForeignPtr (xlaRemove (fromIntegral 0))) t'
  return $ zipWith Device t'' size

run a = head $ runInner [a]

evalAll :: [Result] -> [TreeRaw] -> State Int [Result]
evalAll res [] = return res
evalAll res (x:xs) = do
  x' <- eval res x
  let res' = res ++ [x']
  evalAll res' xs

evalTop :: [TreeRaw] -> (String, [Size])
evalTop trees = (unlines $ [
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
  concat results ++
  ["  ROOT tuple." ++ (show st) ++ " = (" ++ sizes' ++ ") tuple(" ++ locs' ++ ")",
  "}"], sizes)
  where (triples, st) = (runState (evalAll [] trees) 1) :: ([([String], String, Size)], Int)
        sizes = map (\(a,b,c) -> c) triples
        results = map (\(a,b,c) -> a) triples
        locs = map (\(a,b,c) -> b) triples
        sizes' = intercalate ", " (map showSize sizes)
        locs' = intercalate ", " locs

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
len (Device _ size) = len' size

len' (Dim1 m) = m

rows :: Buffer -> Int
rows (Device _ size) = rows' size

rows' (Dim2 m _) = m
rows' (Dim2R m _) = m

cols :: Buffer -> Int
cols (Device _ size) = cols' size

cols' (Dim2 _ n) = n
cols' (Dim2R _ n) = n

--shouldNotBeReachable = error "should not be reachable"

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

{-findSize :: Tree Buffer -> Size
findSize (Pure a) = getSize a
findSize (Impure a) = findSize' $ fmap findSize a-}

findSize' :: [Result] -> TreePrim Size -> Size
findSize' _ (Constant _) = Dim0
findSize' _ (Identity n) = Dim2 n n
findSize' t (UnaryEval (Broadcast s) _) = extractFromTreeSize t s
findSize' t (UnaryEval (Reshape s) _) = extractFromTreeSize t s
findSize' _ (UnaryEval Transpose size) = 
  case size of
    Dim2 a b -> Dim2R b a
    Dim2R a b -> Dim2 b a
    _ -> error "Size Mismatch"
findSize' _ (UnaryEval ReduceArgMax size) = 
  case norm size of 
    Dim2 rows cols -> Dim1 rows
    _ -> error "Size Mismatch"
findSize' _ (UnaryEval op a) = a
findSize' _ (BinEval Gather a b) =
  case (a, b) of
    (Dim2 m 1, Dim2 _ n) -> Dim2 m n
    _ -> error "Size Mismatch"
findSize' _ (BinEval Dot a b) = a `compat` b
  where compat m n = compat' (norm m) (norm n)
        compat' (Dim2 m1 m2) (Dim2 n1 n2)
          | m2 == n1 = Dim2 m1 n2
        compat' (Dim1 m) (Dim1 n)
          | m == n = Dim0
        compat' m n = error $ "Size Mismatch, " ++ show m ++ ", " ++ show n
findSize' _ (BinEval ReduceAdd a b)
  | b == Dim0 = case norm a of
                  Dim1 m -> Dim0
                  Dim2 m n -> Dim1 m
                  _ -> error "Size Mismatch"
  | otherwise = error "Size Mismatch"
findSize' _ (BinEval op a b)
  | a `compat` b = a
  | otherwise = error "Size Mismatch"

type Result = ([String], String, Size)

eval :: [Result] -> TreeRaw -> State Int Result
eval a (Impure (CrossRef n)) = do
  let (_, c, d) = a !! (length a - n)
  return ([], c, d)

eval _ (Pure (i, size)) = 
  complete size "parameter" [] [show i] []

eval t (Impure (UnaryEval (Broadcast size'') tree)) = do
  (result, loc, size') <- eval t tree
  let size = extractFromTreeSize t size''
  let args = compat size' size
  complete size "broadcast" result [loc] args
  where compat Dim0 (Dim2R _ _) = error "Size Mismatch"
        compat Dim0 _ = [("dimensions", "{}")]
        compat (Dim1 m) (Dim2 n1 n2)
          | m == n1 && m == n2 = error "Ambiguous"
          | m == n2 = [("dimensions", "{1}")]
          | m == n1 = [("dimensions", "{0}")]
        compat m n = error $ "Size Mismatch " ++ show m ++ ", " ++ show n

eval t (Impure (UnaryEval Transpose tree)) = do
  (result, loc, size) <- eval t tree
  let outputSize = findSize' t (UnaryEval Transpose size) 
  complete outputSize "transpose" result [loc] [("dimensions", "{1,0}")]

eval t (Impure (UnaryEval ReduceArgMax tree)) = do
  (result, loc, size) <- eval t tree
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

eval t (Impure (UnaryEval op tree)) = do
  (result, loc, size) <- eval t tree
  let opStr = unary op
  let outputSize = findSize' t (UnaryEval op size)
  complete outputSize opStr result [loc] []

eval t (Impure (BinEval (Compare op) treeA treeB)) = do
  (result, loc, size) <- eval t treeA
  (result', loc', size') <- eval t treeB
  let outputSize = findSize' t (BinEval (Compare op) size size')
  st <- get
  let compare = "  compare." ++ show st ++ " = " ++ showSize' "pred" outputSize ++ " compare(" ++ loc ++ ", " ++ loc' ++ "), direction=" ++ show op 
  put (st + 1)
  complete outputSize "convert" (result ++ result' ++ [compare]) ["compare." ++ show st] []

eval t m@(Impure (BinEval Dot treeA treeB)) = do
  (result, loc, size) <- eval t treeA
  (result', loc', size') <- eval t treeB
  let outputSize = findSize' t (BinEval Dot size size')
  let args = case outputSize of
               Dim0 -> [("lhs_contracting_dims", "{0}"), ("rhs_contracting_dims", "{0}")]
               Dim2 _ _ -> [("lhs_contracting_dims", "{1}"), ("rhs_contracting_dims", "{0}")]
  complete outputSize "dot" (result ++ result') [loc, loc'] args

eval t (Impure (BinEval Gather treeA treeB)) = do
  (result, loc, size) <- eval t treeA
  (result', loc', size') <- eval t treeB
  let outputSize@(Dim2 rows cols) = findSize' t (BinEval Gather size' size)
  st <- get
  let convert = "  convert." ++ show st ++ " = s32[" ++ show rows ++ ",1]{1,0} convert(" ++ loc' ++ ")" 
  put (st + 1)
  complete outputSize "gather" (result ++ result' ++ [convert]) [loc, "convert." ++ show st] 
    [("offset_dims", "{1}"), ("collapsed_slice_dims", "{0}"), ("start_index_map","{0}"),
      ("index_vector_dim", "1"), ("slice_sizes", "{1," ++ show cols ++ "}")]

eval t (Impure (BinEval ReduceAdd treeA treeB)) = do
  (result, loc, size) <- eval t treeA
  (result', loc', size') <- eval t treeB
  let outputSize = findSize' t (BinEval ReduceAdd size size')
  let dims = case outputSize of
               Dim1 _ -> "{1}"
               Dim0 -> "{0}"
  complete outputSize "reduce" (result ++ result') [loc, loc'] [("dimensions", dims), ("to_apply", "primitive_computation_add.0")]

eval t (Impure (BinEval op treeA treeB)) = do
  (result, loc, size) <- eval t treeA
  (result', loc', size') <- eval t treeB
  let outputSize = findSize' t (BinEval op size size')
  let opStr = fromJust $ lookup op binary
  complete outputSize opStr (result ++ result') [loc, loc'] []

eval t (Impure (Constant f)) = do
  let outputSize = findSize' t (Constant f)
  complete outputSize "constant" [] [show f] []

eval _ (Impure (Identity n)) = do
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

binEval a b c = Impure $ BinEval a b c
unaryEval a b = Impure $ UnaryEval a b

{-# INLINE[1] binEval #-}
{-# INLINE[1] unaryEval #-}
--{-# INLINE CONLIKE[1] run #-}
-- {-# NOINLINE runInner #-}
{-# NOINLINE run #-}
{-# NOINLINE rows #-}
{-# NOINLINE cols #-}
{-# NOINLINE len #-}
{-# INLINE square #-}
{-# INLINE (@@) #-}
{-# INLINE reshape #-}
{-# INLINE gather #-}
{-# INLINE broadcast' #-}
{-# INLINE broadcast #-}
{-# INLINE sum #-}
{-# INLINE eq #-}
{-# INLINE softmax #-}
{-# INLINE mean #-}
{-# INLINE identity #-}
{-# INLINE transpose #-}
{-# INLINE reduceArgMax #-}
-- {-# INLINE CONLIKE [1] run #-}

{-# RULES "hashan/binEvalRight" forall x y z.
      binEval x y (Pure (run z)) = binEval x y z ; #-}
{-# RULES "hashan/binEvalLeft" forall x y z.
      binEval x (Pure (run y)) z = binEval x y z ; #-}
{-# RULES "hashan/unaryEval" forall x y.
      unaryEval x (Pure (run y)) = unaryEval x y ; #-}

{- {-# RULES "hashan/rows" forall x.
      rows (run x) = rows' (findSize x) ; #-}
{-# RULES "hashan/cols" forall x.
      cols (run x) = cols' (findSize x) ; #-}
{-# RULES "hashan/len" forall x.
      len (run x) = len' (findSize x) ; #-} -}

{-- {-# RULES "hashan/binEvalRight" forall x y z.
      run (BinEval x y (Parameter (run z))) = run (BinEval x y z) ; #-}
{-# RULES "hashan/binEvalLeft" forall x y z.
      run (BinEval x (Parameter (run y)) z) = run (BinEval x y z) ; #-}
{-# RULES "hashan/unaryEval" forall x y.
      run (UnaryEval x (Parameter (run y))) = run (UnaryEval x y) ; #-} --}

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

compute :: (Pass n, VecHelpers n) => Vec n Layer -> Code (Vec n (Tensor, Tensor) -> Tensor -> Tensor -> (Vec n (Tensor, Tensor), Tensor))
compute layers
  = [|| \weights targets input -> 
      let activations = $$(forwardPass layers) weights input in
      let preds = softmax (last' activations) in
      let !loss = negate (mean (sum (log preds * targets))) in
      let delta = (preds - targets) / (broadcast' (rows input, 10) (c (rows input))) in
      let !backward = $$(backwardPass (reverse' layers)) (reverse' weights) (reverse' activations) delta in
      (backward, loss)
    ||]

class Pass n where
  forwardPass :: Vec n Layer -> Code (Vec n (Tensor, Tensor) -> Tensor -> Vec (S n) Tensor)
  backwardPass :: Vec n Layer -> Code (Vec n (Tensor, Tensor) -> Vec (S n) Tensor -> Tensor -> Vec n (Tensor, Tensor))

instance Pass Z where
  forwardPass (End (l1, l2)) 
    = [|| \(End (x1, x2)) y -> 
          let tmp = $$l1 (x1, x2, y) in
          y `cons` nil tmp
      ||]
  backwardPass (End (l1, l2)) 
    = [|| \(End (x1, x2)) (i1, End i2) delta -> 
          let (!a, !b, !c) = $$l2 (x1, x2, i2) i1 delta in
          nil (a, b)
      ||]

instance Pass n => Pass (S n) where
  forwardPass ((l1, l2), ls) 
    = [|| \((x1, x2), xs) y ->
        let tmp = $$l1 (x1, x2, y) in
        y `cons` $$(forwardPass ls) xs tmp
      ||]
  backwardPass ((l1, l2), ls)
    = [|| \((x1, x2), xs) (i1, (i2, is)) delta ->
          let (!a, !b, !c) = $$l2 (x1, x2, i2) i1 delta in
          (a, b) `cons` $$(backwardPass ls) xs (i2 `cons` is) c
      ||]

layers = linearTanh `cons` (linearTanh `cons` (nil linear))

transpose a = run (unaryEval Transpose (Pure a))
reduceArgMax a = run (unaryEval ReduceArgMax (Pure a))

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

identity n = run (Impure (Identity n))
square x = x * x
broadcast s =  run . unaryEval (Broadcast (makeIntoTreeSize (Dim1 s))) . Pure
broadcast' (m, n) =  run . unaryEval (Broadcast (makeIntoTreeSize (Dim2 m n))) . Pure

linear :: Layer
linear = (
    [|| \(w, b, x) -> x @@ w + (broadcast' (rows x, cols w) b) ||], 
    [|| \(w, b, x) _ deltaZ -> 
        let deltaW = transpose x @@ deltaZ in
        let deltaB = (broadcast' (1, rows x) 1) @@ deltaZ in
        let deltaX = deltaZ @@ transpose w in
        (deltaW, reshape deltaB (Dim1 (cols deltaB)), deltaX) ||]
  )

sum x = run (binEval ReduceAdd (Pure x) (Pure 0))
mean x = sum x / c (len x)
softmax x = exp x / (broadcast' (rows x, cols x) (sum (exp x)))

data Nat = Z | S Nat

--infixr 5 :-

cons :: t -> Vec n t -> Vec (S n) t
cons a b = (a, b)

nil :: t -> Vec Z t
nil x = End x

newtype End a = End a

type Vec :: Nat -> Type -> Type
type family Vec n a = result | result -> a n where
  Vec Z     a = End a
  Vec (S n) a = (a, Vec n a)

class VecHelpers n where
  mapM' :: (Monad m) => (t1 -> m t2) -> Vec n t1 -> m (Vec n t2)
  zipWith' :: (t1 -> t2 -> t3) -> Vec n t1 -> Vec n t2 -> Vec n t3
  snoc' :: Vec n t -> t -> Vec (S n) t
  reverse' :: Vec n t -> Vec n t
  length' :: Vec n t -> Int
  last' :: Vec n t -> t

instance VecHelpers Z where
  mapM' f (End a) = End <$> f a
  zipWith' f (End a) (End b) = nil (f a b)
  snoc' (End a) x = a `cons` nil x
  reverse' (End a) = nil a
  length' (End a) = 1
  last' (End a) = a

instance VecHelpers n => VecHelpers (S n) where
  mapM' f (a, b) = liftM2 cons (f a) (mapM' f b)
  zipWith' f (a1, b1) (a2, b2) = f a1 a2 `cons` zipWith' f b1 b2
  snoc' (x, y) z = x `cons` snoc' y z
  reverse' (x, y) = snoc' (reverse' y) x
  length' (x, y) = 1 + length' y
  last' (x, y) = last' y

{-type Vec :: Nat -> Type -> Type
data Vec t n where
  Nil :: Vec Z t
  (:-) :: !t -> !(Vec n t) -> Vec (S n) t-}

{-mapM' :: (Monad m) => (t1 -> m t2) -> Vec n t1 -> m (Vec n t2)
mapM' f Nil = return Nil
mapM' f (a :- b) = liftM2 (:-) (f a) (mapM' f b)

zipWith' :: (t1 -> t2 -> t3) -> Vec n t1 -> Vec n t2 -> Vec n t3
zipWith' f Nil Nil = Nil
zipWith' f (a1 :- b1) (a2 :- b2) = f a1 a2 :- zipWith' f b1 b2

snoc' :: Vec n t -> t -> Vec (S n) t
snoc' Nil x = x :- Nil
snoc' (x :- y) z = x :- snoc' y z

reverse' :: Vec n t -> Vec n t
reverse' Nil = Nil
reverse' (x :- y) = snoc' (reverse' y) x

last' :: Vec n t -> t
last' (x :- Nil) = x
last' (x :- y) = last' y

length' :: Vec n t -> Int
length' Nil = 0
length' (x :- y) = 1 + length' y-}

