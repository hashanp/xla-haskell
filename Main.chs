{-# LANGUAGE ForeignFunctionInterface, TypeApplications, FlexibleInstances, TemplateHaskell #-}
module Main where

import Prelude hiding (sin)
import Lib
import Foreign.C.String
import Foreign.C
import Foreign.Ptr (Ptr, nullPtr)
import Data.Maybe (fromJust)
import System.Mem
import Foreign.Marshal.Array
import Foreign.Marshal.Alloc
import System.IO.Unsafe (unsafePerformIO)

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

f = do
  initialise
  let a = make' [1, 2, 3, 4]
  let b = Parameter (run (unaryEval (Broadcast (Dim2 4 4)) a))
  --let c = a + b
  return (b, b)
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

main = do
  --let hlo = evalTop (BinEval Add (BinEval Power (Broadcast 2) (Parameter 0)) (Parameter 1))
  --putStrLn hlo 
  --code <- newCString hlo -- =<< readFile "/tmp/fn_hlo.txt"
  (Parameter x, Parameter y) <- f
  print (x, y)
  printView x
  print "end"
