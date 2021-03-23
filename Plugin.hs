{-# LANGUAGE TupleSections, BlockArguments #-}

module Plugin (plugin) where
import GhcPlugins
import Outputable (showSDocUnsafe, interpp'SP)
import Module (Module, mkModuleName)
import Finder (findImportedModule)
import IfaceEnv (lookupOrigIO)
import OccName hiding (varName, mkTcOcc) -- (mkVarOcc, mkDataOcc)
import Data.Generics hiding (TyCon)
import Control.Monad (when)
import TysWiredIn (consDataCon, consDataConName, nilDataConName, nilDataCon)
import PrelNames (metaConsDataConName)
import Name hiding (varName)
import TyCon (mkPrelTyConRepName, TyCon)
import CoreFVs (exprFreeVars)
import VarSet (VarSet, elemVarSet)
import Debug.Trace (trace)
import HscTypes (lookupTyCon)

plugin :: Plugin
plugin = defaultPlugin 
  { installCoreToDos = install
  , pluginRecompile = purePlugin
  }

data State = State { getRun :: Name, getTree :: TyCon, getBuffer :: TyCon }

transformProgram :: ModGuts -> CoreM ModGuts
transformProgram guts = do
  hscEnv <- getHscEnv
  mod' <- liftIO $ findImportedModule hscEnv (mkModuleName "Lib") Nothing
  --mod'' <- liftIO $ findImportedModule hscEnv (mkModuleName "GHC.Types") Nothing
  let (Found _ mod) = mod'
  name <- liftIO $ lookupOrigIO hscEnv mod (mkVarOcc "runInner")
  name' <- liftIO $ lookupOrigIO hscEnv mod (mkTcOcc "Tree")
  name'' <- liftIO $ lookupOrigIO hscEnv mod (mkTcOcc "Buffer")
  tyCon <- lookupTyCon name'
  tyCon' <- lookupTyCon name''
  putMsgS (showSDocUnsafe $ ppr name)
  newBinds <- mapM (transformFunc (State { getRun = name, getTree = tyCon, getBuffer = tyCon' }) guts) (mg_binds guts)
  return $ guts { mg_binds = newBinds }

transformFunc :: State -> ModGuts -> CoreBind -> CoreM CoreBind
transformFunc st guts x = do
  putMsgS "---"
  putMsgS $ showSDocUnsafe $ interpp'SP $ getNames x
  b <- shouldTransformBind guts x
  if b
    then transformBind st x -- everywhereM (mkM (transformExpr st)) x -- mkM/everywhereM are from 'syb'
    else return x

shouldTransformBind guts _ = return True

getNames (NonRec b _) = [b]
getNames (Rec bs) = map (\(a, b) -> a) bs 

matchesRun st (Var x) = varName x == getRun st
matchesRun _ _ = False

matchesNil (Var x) = x == dataConWorkId nilDataCon
matchesNil _ = False

matchesCons (Var x) = x == dataConWorkId consDataCon
matchesCons _ = False

{-putMsgS (showSDocUnsafe (ppr (nameUnique (varName x)))) -- return () --[e2]
                       putMsgS (showSDocUnsafe (ppr (nameUnique (dataConName (consDataCon)))))
                       putMsgS (showSDocUnsafe (ppr (nameUnique (consDataConName))))
                       putMsgS (showSDocUnsafe (ppr (nameUnique (varName (dataConWorkId consDataCon)))))
                       putMsgS (showSDocUnsafe (ppr (nameUnique (mkPrelTyConRepName (consDataConName)))))-}

convert :: State -> CoreExpr -> Maybe [CoreExpr] -- [CoreExpr]
convert _ e@(App nil (Type _))
  | matchesNil nil = return []
convert st e@(App (App (App cons (Type ty)) e1) e2) 
  | matchesCons cons = do 
      trace (showSDocUnsafe (ppr (tyConAppTyCon (tyConAppArgs ty !! 0) == getTree st))) (return ())
      rest <- convert st e2 --  && matchesNil l2 = putMsgS "success"
      return $ e1:rest
convert st e = Nothing
  --error "here"

data RunExpr = RunExpr CoreExpr [CoreExpr] VarSet
type Info = [RunExpr]

notPresent x (RunExpr _ _ varSet) = not $ x `elemVarSet` varSet

transformBind :: State -> CoreBind -> CoreM CoreBind
transformBind st (NonRec b e) = (NonRec b) <$> (fst <$> transformExpr st e)
transformBind st (Rec bs) = Rec <$> (mapM (\(b, e) -> (b,) <$> fst <$> transformExpr st e) bs)

transformExpr :: State -> CoreExpr -> CoreM (CoreExpr, Info)

transformExpr st e@(App e1 e2)
  | matchesRun st e1 = do
        case convert st e2 of
            Nothing -> do
                putMsgS "fail"
                return (e, [])
            Just a -> do
                putMsgS "success"
                return (e, [RunExpr e a (exprFreeVars e)])
  | otherwise = do
        (e1', info1) <- transformExpr st e1
        (e2', info2) <- transformExpr st e2
        let e' = App e1' e2'
        when ((not (null info1)) && (not (null info2))) do
          putMsgS "join"
        return (e', info1 ++ info2)

transformExpr st e@(Lam x e1) = do
  (e1', info) <- transformExpr st e1
  let e' = Lam x e1'
  let info' = filter (notPresent x) info
  when (length info /= length info') do
    putMsgS "stop"
  return (e', info')

{-transformExpr st e@(Let (NonRec x e1) e2) = do
  (e1', info1) <- transformExpr st e1
  (e2', info2) <- transformExpr st e2
  let e' = Let (NonRec x e1') e2'
  let info1' = info1
  let info2' = filter (notPresent x) info2
  return (e', info1' ++ info2')-}

transformExpr st e = return (e, [])

{-transformExpr :: State -> CoreExpr -> CoreM CoreExpr
-- See 'Id'/'Var' in 'compiler/basicTypes/Var.lhs' (note: it's opaque)
transformExpr st e@(Var x) | isTyVar x    = return e
                        | isTcTyVar x  = return e
                        | isLocalId x  = return e
                        | isGlobalId x = return e
-- See 'Literal' in 'compiler/basicTypes/Literal.lhs'
transformExpr st e@(Lit l)     = return e
transformExpr st e@(App e1 e2) = do when (matchesRun st e1) (convert st e2)
                                    return e
transformExpr st e@(Lam x e1)   = return e
-- b is a Bind CoreBndr, which is the same as CoreBind
transformExpr st e@(Let b e1)   = return e
-- Remember case in core is strict!
transformExpr st e@(Case e1 b t as) = return e
-- XXX These are pretty esoteric...
transformExpr _ e@(Cast e1 c)  = return e
transformExpr _ e@(Tick t e1)  = return e
transformExpr _ e@(Type t)    = return e
transformExpr _ e@(Coercion c) = return e-}

install :: [CommandLineOption] -> [CoreToDo] -> CoreM [CoreToDo]
install _ todo = do
  putMsgS "Hello!"
  let pass = CoreDoPluginPass "Template" transformProgram
  return $ todo ++ [pass]