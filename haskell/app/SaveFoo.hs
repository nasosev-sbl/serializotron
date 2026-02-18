module Main where

import GHC.Generics
import Serializotron

data Foo = A Int | B Int
  deriving stock (Generic, Show, Eq)
  deriving anyclass (ToSZT, FromSZT)

main :: IO ()
main = do
  let value = A 42 :: Foo
  putStrLn $ "Saving: " ++ show value
  saveSzt "foo.szt" value
  putStrLn "Saved to foo.szt"
