module Plugin (plugin) where
import GhcPlugins
import Outputable (showSDocUnsafe, interpp'SP)
import Module (Module, mkModuleName)
import Finder (findImportedModule)
import IfaceEnv (lookupOrigIO)
import OccName hiding (varName) -- (mkVarOcc, mkDataOcc)
import Data.Generics
import Control.Monad (when)
import TysWiredIn (consDataCon, consDataConName, nilDataConName, nilDataCon)
import PrelNames (metaConsDataConName)
import Name hiding (varName)
import TyCon (mkPrelTyConRepName)

plugin :: Plugin
plugin = defaultPlugin 
  { installCoreToDos = install
  , pluginRecompile = purePlugin
  }

data State = State { getRun :: Name, getCons :: Name }

transformProgram :: ModGuts -> CoreM ModGuts
transformProgram guts = do
  hscEnv <- getHscEnv
  mod' <- liftIO $ findImportedModule hscEnv (mkModuleName "Lib") Nothing
  mod'' <- liftIO $ findImportedModule hscEnv (mkModuleName "GHC.Types") Nothing
  let (Found _ mod) = mod'
  name <- liftIO $ lookupOrigIO hscEnv mod (mkVarOcc "runInner")
  name' <- liftIO $ lookupOrigIO hscEnv mod (mkVarOcc ":")
  putMsgS (showSDocUnsafe $ ppr name)
  newBinds <- mapM (transformFunc (State { getRun = name, getCons = name' }) guts) (mg_binds guts)
  return $ guts { mg_binds = newBinds }

transformFunc :: State -> ModGuts -> CoreBind -> CoreM CoreBind
transformFunc st guts x = do
  putMsgS "---"
  putMsgS $ showSDocUnsafe $ interpp'SP $ getNames x
  b <- shouldTransformBind guts x
  if b
    then everywhereM (mkM (transformExpr st)) x -- mkM/everywhereM are from 'syb'
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

convert :: State -> CoreExpr -> CoreM () -- [CoreExpr]
convert _ e@(App (App (App l (Type _)) e3) (App l2 (Type _))) 
  | matchesCons l && matchesNil l2 = putMsgS "success"
convert _ e = do
  putMsgS "fail"
  putMsgS (showSDocUnsafe (ppr e)) 
  --error "here"

transformExpr :: State -> CoreExpr -> CoreM CoreExpr
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
transformExpr _ e@(Coercion c) = return e

install :: [CommandLineOption] -> [CoreToDo] -> CoreM [CoreToDo]
install _ todo = do
  putMsgS "Hello!"
  let pass = CoreDoPluginPass "Template" transformProgram
  return $ todo ++ [pass]