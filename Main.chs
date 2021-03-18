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
c' = Constant
broadcast s = Parameter . run . unaryEval (Broadcast (Dim1 s))

sum x = Parameter (run (binEval ReduceAdd x 0))
mean x = sum x / c (len x)
softmax x = exp x / (broadcast (len x) (sum (exp x)))
square x = x * x

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
  return xs

makeR (m, n) = do
  let len = m * n
  values <- getRandoms
  return (create2D (m, n) (take len values))

makeR' m = do
  values <- getRandoms
  return (create1D (take m values))

type Code a = TExpQ a
type Tensor = Tree Buffer
type Layer = (Code ((Tensor, Tensor, Tensor) -> Tensor), Code ((Tensor, Tensor, Tensor) -> Tensor -> Tensor -> (Tensor, Tensor, Tensor)))

transpose a = Parameter (run (unaryEval Transpose a))

linearTanh :: Layer
linearTanh = (
    [|| \(w, b, x) -> tanh (x @@ w + b) ||], 
    [|| \(w, b, x) a deltaA -> 
        let deltaZ = deltaA * (1 - square a) in
        let deltaW = transpose x @@ deltaZ in
        let deltaB = 1 @@ deltaZ in
        let deltaX = deltaZ @@ transpose w in
        (deltaW, deltaB, deltaX) ||]
  )

linear :: Layer
linear = (
    [|| \(w, b, x) -> x @@ w + b ||], 
    [|| \(w, b, x) _ deltaZ -> 
        let deltaW = transpose x @@ deltaZ in
        let deltaB = 1 @@ deltaZ in
        let deltaX = deltaZ @@ transpose w in
        (deltaW, deltaB, deltaX) ||]
  )

forwardPass :: [Layer] -> Code ([(Tensor, Tensor)] -> Tensor -> [Tensor])
forwardPass [] = [|| \[] _ -> [] ||]
forwardPass ((l1, l2):ls) 
  = [|| \((x1, x2):xs) y ->
      let tmp = $$l1 (x1, x2, y) in
      tmp : $$(forwardPass ls) xs tmp
    ||]

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