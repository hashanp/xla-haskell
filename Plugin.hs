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
import Type (mkTyConApp)
import FastString (fsLit)
import CoreUtils (exprType)
import CoreMonad (SimplMode)
import BasicTypes (CompilerPhase(..))

plugin :: Plugin
plugin = defaultPlugin 
  { installCoreToDos = install
  , pluginRecompile = purePlugin
  }

data State = State { getRealRun :: Name, getRealRunId :: Id, getRun :: Name, getRunId :: Id, getTree :: TyCon, getBuffer :: TyCon, getError :: Id }

transformProgram :: ModGuts -> CoreM ModGuts
transformProgram guts = do
  hscEnv <- getHscEnv
  mod' <- liftIO $ findImportedModule hscEnv (mkModuleName "Lib") Nothing
  --mod'' <- liftIO $ findImportedModule hscEnv (mkModuleName "GHC.Types") Nothing
  let (Found _ mod) = mod'
  realRun <- liftIO $ lookupOrigIO hscEnv mod (mkVarOcc "run")
  name <- liftIO $ lookupOrigIO hscEnv mod (mkVarOcc "runInner")
  name' <- liftIO $ lookupOrigIO hscEnv mod (mkTcOcc "Tree")
  name'' <- liftIO $ lookupOrigIO hscEnv mod (mkTcOcc "Buffer")
  name''' <- liftIO $ lookupOrigIO hscEnv mod (mkVarOcc "shouldNotBeReachable")
  tyCon <- lookupTyCon name'
  tyCon' <- lookupTyCon name''
  id <- lookupId name'''
  runId <- lookupId name
  realRunId <- lookupId realRun
  putMsgS (showSDocUnsafe $ ppr name)
  newBinds <- mapM (transformFunc (State { getRealRun = realRun, getRealRunId = realRunId, getRun = name, getRunId = runId, getTree = tyCon, getBuffer = tyCon', getError = id }) guts) (mg_binds guts)
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

matchesRun st (Var x) = varName x == getRealRun st
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

data RunExpr = RunExpr Var CoreExpr VarSet
type Info = [RunExpr]

present xs (RunExpr _ _ varSet) = any (`elemVarSet` varSet) xs

getType :: State -> CoreExpr
getType st = Type $ mkTyConApp (getTree st) [mkTyConApp (getBuffer st) []]

getType' :: State -> Type
getType' st = mkTyConApp listTyCon [mkTyConApp (getBuffer st) []]

getType'' :: State -> Type
getType'' st = mkTyConApp (getBuffer st) []

createRun :: State -> [CoreExpr] -> CoreM CoreExpr
createRun st [] = do
  return $ App (Var (dataConWorkId nilDataCon)) (getType st)
createRun st (x:xs) = do
  rest <- createRun st xs
  return $ App (App (App (Var (dataConWorkId consDataCon)) (getType st)) x) rest

names' :: State -> [CoreExpr] -> CoreM [Var]
names' _ [] = return []
names' st (x:xs) = do
  xs' <- names' st xs
  var2 <- mkSysLocalM (fsLit "hashan2") (getType'' st)
  return $ var2:xs'

getAll :: Info -> ([Var], [CoreExpr])
getAll [] = ([], [])
getAll (RunExpr ns ls _:xs) = (ns:ns', ls:ls')
  where (ns', ls') = getAll xs

createDown :: State -> Type -> [Var] -> CoreExpr -> CoreExpr -> CoreM CoreExpr
createDown _ _ [] _ e' = return e'
createDown st ty (var2:ns) e e' = do
  var1 <- mkSysLocalM (fsLit "hashan1") (getType' st)
  var3 <- mkSysLocalM (fsLit "hashan3") (getType' st)
  rest <- createDown st ty ns (Var var3) e'
  return $ Case e var1 ty 
    [(DataAlt consDataCon, [var2, var3], rest),
     (DataAlt nilDataCon, [], Var (getError st))]

make :: State -> Info -> CoreExpr -> CoreM CoreExpr
make st info e = do
  let (ns, ls) = getAll info
  run <- createRun st ls
  let expr = App (Var (getRunId st)) run
  createDown st (exprType e) ns expr e

{-
names :: Info -> [(CoreExpr, [Name])]
names = mapM (\(RunExpr a b _) -> (a,) <$> )-}

{--}

transformBind :: State -> CoreBind -> CoreM CoreBind
transformBind st (NonRec b e) = (NonRec b) <$> (transformExpr' st e)
transformBind st (Rec bs) = Rec <$> (mapM (\(b, e) -> (b,) <$> transformExpr' st e) bs)

transformExpr' :: State -> CoreExpr -> CoreM CoreExpr
transformExpr' st e = do
  (e', info) <- transformExpr st e
  force st info e'

force :: State -> Info -> CoreExpr -> CoreM CoreExpr
force _ [] expr = return expr
force st [RunExpr x e varSet] expr = return $ Let (NonRec x (App (Var (getRealRunId st)) e)) expr
force st info expr = make st info expr

transformExpr :: State -> CoreExpr -> CoreM (CoreExpr, Info)

transformExpr st e@(App e1 e2)
  | matchesRun st e1 = do
        {- case convert st e2 of
            Nothing -> do
                putMsgS "fail"
                return (e, [])
            Just a -> do
                --names <- names' st a-}
        putMsgS "success"
        var2 <- mkSysLocalM (fsLit "hashan2") (getType'' st)
        --run <- createRun st (map Var names)
        return (Var var2, [RunExpr var2 e2 (exprFreeVars e2)])
  | otherwise = do
        (e1', info1) <- transformExpr st e1
        (e2', info2) <- transformExpr st e2
        let e' = App e1' e2'
        {-when ((not (null info1)) && (not (null info2))) do
          putMsgS "join"-}
        {-if ((not (null info1)) && (not (null info2)))
          then do
            ee <- make st (info1 ++ info2) e'
            return (ee, [])
          else return (e', info1 ++ info2)-}
        return (e', info1 ++ info2)

transformExpr st e@(Lam x e1) = do
  (e1', info) <- transformExpr st e1
  --let info' = filter (notPresent x) info
  {-if (length info /= length info')
    then do -}
  e1'' <- force st info e1'
  return $ (Lam x e1'', [])
    {-else
      return (Lam x e', info')-}

transformExpr st e@(Let (NonRec x e1) e2) = do
  --(e1', info1) <- transformExpr st e1
  putMsgS "let"
  (e1', info1) <- transformExpr st e1
  (e2', info2) <- transformExpr st e2
  if not (null (filter (present [x]) info2))
    then do
      e2'' <- force st info2 e2'
      return (Let (NonRec x e1') e2'', info1)
    else
      return (Let (NonRec x e1') e2', info1 ++ info2)
  
transformExpr st (Case e1 b t as) = do
  (e1', info1) <- transformExpr st e1
  (as', info2) <- transformAlts st as
  return (Case e1' b t as', info1 ++ info2)

transformExpr st e = return (e, [])

transformAlts st [(f, h, g)] = do
  putMsgS "caseOne"
  (g', info) <- transformExpr st g
  if not (null (filter (present h) info))
    then do
      g'' <- force st info g'
      return ([(f, h, g'')], [])
    else
      return ([(f, h, g')], info)

transformAlts st as = do
  putMsgS "case"
  as' <- mapM (\(a, b, c) -> ((a,b,) <$> transformExpr' st c)) as
  return $ (as', [])

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
  dflags <- getDynFlags
  let simplPass = SimplMode 
                    { sm_phase      = Phase 0
                    , sm_names      = []
                    , sm_dflags     = dflags
                    , sm_rules      = False
                    , sm_eta_expand = True
                    , sm_inline     = True
                    , sm_case_case  = True
                    }
  let pass = CoreDoPluginPass "Template" transformProgram
  return $ todo ++ [pass, CoreDoSimplify 3 simplPass]