module Plugin (plugin) where
import GhcPlugins
import Outputable (showSDocUnsafe, interpp'SP)
import Module (Module, mkModuleName)
import Finder (findImportedModule)
import IfaceEnv (lookupOrigIO)
import OccName (mkVarOcc)
import Data.Generics
import Control.Monad (when)

plugin :: Plugin
plugin = defaultPlugin 
  { installCoreToDos = install
  , pluginRecompile = purePlugin
  }

data State = State { getRun :: Name }

transformProgram :: ModGuts -> CoreM ModGuts
transformProgram guts = do
  hscEnv <- getHscEnv
  mod' <- liftIO $ findImportedModule hscEnv (mkModuleName "Lib") Nothing
  let (Found _ mod) = mod'
  name <- liftIO $ lookupOrigIO hscEnv mod (mkVarOcc "run")
  putMsgS (showSDocUnsafe $ ppr name)
  newBinds <- mapM (transformFunc (State { getRun = name }) guts) (mg_binds guts)
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

transformExpr :: State -> CoreExpr -> CoreM CoreExpr
-- See 'Id'/'Var' in 'compiler/basicTypes/Var.lhs' (note: it's opaque)
transformExpr st e@(Var x) | isTyVar x    = return e
                        | isTcTyVar x  = return e
                        | isLocalId x  = return e
                        | isGlobalId x = do when (varName x == getRun st) (putMsgS (showSDocUnsafe $ ppr x))
                                            return e
-- See 'Literal' in 'compiler/basicTypes/Literal.lhs'
transformExpr st e@(Lit l)     = return e
transformExpr st e@(App e1 e2) = return e
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