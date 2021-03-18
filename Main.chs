{-# LANGUAGE ForeignFunctionInterface, TypeApplications, FlexibleInstances, TemplateHaskell #-}
module Main where

import Prelude hiding (sin, sum)
import Lib
import Foreign.C.String
import Foreign.C
import Foreign.Ptr (Ptr, nullPtr)
import Data.Maybe (fromJust)
--import System.Mem
import Foreign.Marshal.Array
import Foreign.Marshal.Alloc
import System.IO.Unsafe (unsafePerformIO)
import Data.IDX.Internal
import Data.IDX
import qualified Data.Vector.Unboxed as V
import System.Random

-- pure function
--foreign import ccall "hashan" c_sin :: CDouble -> CDouble
--sin :: Double -> Double
--sin d = realToFrac (c_sin (realToFrac d))

-- impure function
--foreign import ccall "time" c_time :: Ptr a -> IO CTime
--getTime :: IO CTime
--getTime = c_time nullPtr

make :: [[Float]] -> Tree Buffer
make a = Parameter $ create2D (length a, length (head a)) (concat a)

make' :: [Float] -> Tree Buffer
make' = Parameter . create1D

c = Constant . fromIntegral
broadcast s = Parameter . run . unaryEval (Broadcast (Dim1 s))

sum x = Parameter (run (binEval ReduceAdd x 0))
mean x = sum x / c (len x)
softmax x = exp x / (broadcast (len x) (sum (exp x)))

f = do
  initialise
  let a = make [[1, 2], [3, 4]]
  let b = make' [1, 2, 3, 4]
  let c = softmax b
  --let c = a + b
  return (c, c)
  {-let a = make [ [1, 1]
               , [2, 1] ]
  let b = make [ [2, 2]
               , [3, 2] ]
  --let c = $$(mul [[||2||], [||2||], [||2||]]) a
  let c = Parameter (run (MatMul a b))
  return (c, c)-}
  
--let b = create (2, 2) [1, 1, 1, 3]
--run (BinEval Add (UnaryEval Negate (Parameter a)) (Parameter b))
--let c = create (2, 2) [1, 1, 1, 1]
--run (BinEval Add (BinEval Add (Parameter b) (Parameter a)) (Parameter a))
--let b = 2 * a
--let a = {# call pure XLA_CreateBuffer as ^ #} 1 7 3 4
--let b = {# call pure XLA_CreateBuffer as ^ #} 1 1 1 3

oneHotEncode x = map (\y -> if x == y then 1 else 0) [0..9]

makeR (m, n) = do
  gen <- newStdGen
  let len = m * n
  return (create2D (m, n) (take len (randoms @Float gen)))

makeR' m = do
  gen <- newStdGen
  return (create1D (take m (randoms @Float gen)))

main = do
  initialise
  Just images <- decodeIDXFile "../data/t10k-images-idx3-ubyte"
  Just (IDXLabels labels) <- decodeIDXLabelsFile "../data/t10k-labels-idx1-ubyte"
  print $ idxDimensions images
  printView =<< (makeR' 4)
  let y = create2D (10000, 10) (concatMap oneHotEncode (V.toList labels))
  let x = create2D (10000, 28 * 28) (map fromIntegral (V.toList (idxIntContent images)))
  print $ getSize x
  print $ getSize y
  
{- (Parameter x, Parameter y) <- f
  print (x, y)
  printView x
  print "end" -}