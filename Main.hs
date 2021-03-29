{-# OPTIONS_GHC -fplugin=Plugin #-}
{-# LANGUAGE TypeApplications, FlexibleInstances, BangPatterns, DataKinds, GADTs, TemplateHaskell, BlockArguments #-}
module Main where

import Prelude hiding (sin, sum)
import Language.Haskell.TH.Lib
import Lib
import Foreign.C.String
import Foreign.C
import Foreign.Ptr (Ptr, nullPtr)
import Data.Maybe (fromJust)
import System.Mem
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
-- foreign import ccall "hashan" c_sin :: CDouble -> CDouble
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
chunksOf n xs
  | length xs1 == 128 = xs1 : chunksOf n xs2
  | otherwise = []
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

{-# INLINE update #-}
update :: Float -> (Tensor, Tensor) -> (Tensor, Tensor) -> (Tensor, Tensor)
update stepSize (w, b) (deltaW, deltaB) = (w - broadcast' (rows w, cols w) (c' stepSize) * deltaW, b - broadcast (len b) (c' stepSize) * deltaB)

testDataSrc = "../data/t10k-images-idx3-ubyte"
testLabelsSrc = "../data/t10k-labels-idx1-ubyte"
trainDataSrc = "../data/train-images-idx3-ubyte"
trainLabelsSrc = "../data/train-labels-idx1-ubyte"
batchSize = 128
stepSize = 0.001

{-
784 -> 1024 -> 1024 -> 10 
-}

get :: (Int, Int) -> IO (Tensor, Tensor)
get (m, n) = do
  weights <- makeR (m, n) 0.1
  biases <- makeR' n 0.1
  return (weights, biases)

type Two = S (S Z) 

doBatch :: Vec Two (Tensor, Tensor) -> Tensor -> Tensor -> [[Float]] -> Int -> IO (Vec Two (Tensor, Tensor))
doBatch weights _ _ [] _ = return weights
doBatch weights x y (b:bs) n = do
  print $ "iteration " ++ show n
  let z = make' b
  let size = len z
  let x' = gather x (reshape z (Dim2 size 1))
  let y' = gather (reshape y (Dim2 60000 1)) (reshape z (Dim2 size 1))
  let y'' = gather (identity 10) (reshape y' (Dim2 size 1))
  let (!p, !loss) = $$(compute layers) weights y'' x'
  printView loss 
  when (n `mod` 50 == 0) do
    performMajorGC
  doBatch (zipWith' (update stepSize) weights (reverse' p)) x y bs (n + 1)

main = do
  initialise
  Just images <- decodeIDXFile trainDataSrc
  Just (IDXLabels labels) <- decodeIDXLabelsFile trainLabelsSrc
  let other = make' [0..9]
  let y = create1D (map fromIntegral (V.toList labels))
  let x = create2D (60000, 28 * 28) (map fromIntegral (V.toList (idxIntContent images)))
  initial <- mapM' get ( (784, 1024) `cons`
                        ((1024, 1024) `cons`
                        nil (1024, 10)))
  batch <- chunksOf batchSize <$> shuffle [0..59999]
  {-let p = $$(forwardPass layers) initial x'
  print $ head batch
  print $ (length p)
  let preds = softmax (last p)
  --let loss = negate (mean (sum (log preds * y')))
  let delta = (preds - y'') / (broadcast' (rows x', 10) (c (rows x')))
  let p'' = $$(backwardPass (reverse layers)) (reverse initial) (reverse p) delta
  --let res = reduceArgMax preds
  --let res2 = eq (reshape res (Dim2 128 1)) y'-}
  p <- doBatch initial x y (take 200 batch) 0
  let res = $$(forwardPass layers) p x
  let res' = eq y (reduceArgMax (softmax (last' res)))
  let res'' = mean res'
  --printView (gather (fst $ p !! 1) (reshape other (Dim2 10 1)))
  --printView (snd $ last p'')
  --print $ (take 50 batch)
  print $ length' p
  printView res'' --(mean (reshape res2 (Dim1 128)))

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