{-# LANGUAGE ForeignFunctionInterface, TypeApplications, FlexibleInstances, TemplateHaskell, BlockArguments #-}
module Main where

import Prelude hiding (sin, sum)
import Language.Haskell.TH.Lib
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
import Control.Monad.Random
import Data.Array.ST
import Control.Monad.ST
import GHC.Arr
import Data.Random.Normal

-- pure function
--foreign import ccall "hashan" c_sin :: CDouble -> CDouble
--sin :: Double -> Double
--sin d = realToFrac (c_sin (realToFrac d))

-- impure function
--foreign import ccall "time" c_time :: Ptr a -> IO CTime
--getTime :: IO CTime
--getTime = c_time nullPtr

make :: [[Float]] -> Buffer
make a = create2D (length a, length (head a)) (concat a)

make' :: [Float] -> Buffer
make' =  create1D

c = Const . fromIntegral
c' = Const

sum x = run (binEval ReduceAdd (Parameter x) (Parameter 0))
mean x = sum x / c (len x)
softmax x = exp x / (broadcast' (rows x, cols x) (sum (exp x)))

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

chunksOf :: Int -> [a] -> [[a]]
chunksOf _ [] = []
chunksOf n xs = xs1 : chunksOf n xs2
  where (xs1, xs2) = splitAt n xs

shuffle xs = do
  let l = length xs
  rands <- forM [0..(l-2)] (\i -> getRandomR (i, l-1))
  let ar = runSTArray do
        ar <- thawSTArray (listArray (0, l-1) xs)
        forM_ (zip [0..] rands) $ \(i, j) -> do
          vi <- readSTArray ar i
          vj <- readSTArray ar j
          writeSTArray ar j vi
          writeSTArray ar i vj
        return ar
  return (elems ar)

makeR (m, n) scale = do
  let len = m * n
  values <- normalsIO
  return (create2D (m, n) (map (* scale) (take len values)))

makeR' m scale = do
  values <- normalsIO
  return (create1D (map (* scale) (take m values)))

backwardPass :: [Layer] -> Code ([(Tensor, Tensor)] -> [Tensor] -> Tensor -> [(Tensor, Tensor)])
backwardPass ((l1, l2):ls)
  = [|| \((x1, x2):xs) (i:is) delta ->
        let (a, b, c) = $$l2 (x1, x2, i) i delta in
        (a, b) : $$(backwardPass ls) xs is c
    ||]

compute layers
  = [|| \weights targets input -> 
      let activations = $$(forwardPass layers) weights input in
      let preds = softmax (last activations) in
      let loss = negate (mean (sum (log preds * targets))) in
      let delta = (preds - targets) / c (rows input) in
      $$(backwardPass layers) weights activations delta
    ||]

update :: Float -> (Tensor, Tensor) -> (Tensor, Tensor) -> (Tensor, Tensor)
update stepSize (w, b) (deltaW, deltaB) = (w - c' stepSize * deltaW, b - c' stepSize * deltaB)

testDataSrc = "../data/t10k-images-idx3-ubyte"
testLabelsSrc = "../data/t10k-labels-idx1-ubyte"
trainDataSrc = "../data/train-images-idx3-ubyte"
trainLabelsSrc = "../data/train-labels-idx1-ubyte"
batchSize = 128

{-
784 -> 1024 -> 1024 -> 10 
-}

get :: (Int, Int) -> IO (Tensor, Tensor)
get (m, n) = do
  weights <- makeR (m, n) 0.1
  biases <- makeR' n 0.1
  return (weights, biases)

main = do
  initialise
  Just images <- decodeIDXFile trainDataSrc
  Just (IDXLabels labels) <- decodeIDXLabelsFile trainLabelsSrc
  let z = make' [0..9]
  let y = create1D (map fromIntegral (V.toList labels))
  let x = create2D (60000, 28 * 28) (map fromIntegral (V.toList (idxIntContent images)))
  initial <- mapM get [ (784, 1024)
                      , (1024, 1024)
                      , (1024, 10)]
  let p = $$(forwardPass layers) initial x
  batch <- chunksOf batchSize <$> shuffle [0..59999]
  print $ head batch
  print $ (length p)
  let res = reduceArgMax (softmax (last p))
  let res2 = eq res y
  --printView (gather (reshape res2 (Dim2 60000 1)) (reshape z (Dim2 10 1)))
  printView (mean res2)

{-print $ idxDimensions images
  let a = make [[1,2],[3,4],[5,6]]
  let b = make' [1, 0]
  let (Parameter c) = gather a (reshape b (Dim2 2 1))
  printView c
  print $ getSize x
  print $ getSize y
-}

{- (Parameter x, Parameter y) <- f
  print (x, y)
  printView x
  print "end" -}