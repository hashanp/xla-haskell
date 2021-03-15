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



f = do
  initialise
  --let a = {# call pure XLA_CreateBuffer as ^ #} 1 7 3 4
  --let b = {# call pure XLA_CreateBuffer as ^ #} 1 1 1 3
  let a = Parameter (create (2, 2) [1, 1, 1, 1])
  --let c = create (2, 2) [1, 1, 1, 1]
  --run (BinEval Add (BinEval Add (Parameter b) (Parameter a)) (Parameter a))
  let b = 2 * a
  let c = $$(power @(Tree Buffer) 4) b
  return (c, c)
  --let b = create (2, 2) [1, 1, 1, 3]
  --run (BinEval Add (UnaryEval Negate (Parameter a)) (Parameter b))

main = do
  --let hlo = evalTop (BinEval Add (BinEval Power (Broadcast 2) (Parameter 0)) (Parameter 1))
  --putStrLn hlo 
  --code <- newCString hlo -- =<< readFile "/tmp/fn_hlo.txt"
  (Parameter x, Parameter y) <- f
  print (x, y)
  printView x
  print "end"
