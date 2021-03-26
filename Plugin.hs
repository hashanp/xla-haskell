{-# LANGUAGE TupleSections, BlockArguments #-}

module Plugin (plugin) where
import GhcPlugins
import Outputable (showSDocUnsafe, interpp'SP)
import Module (Module, mkModuleName)
import Finder (findImportedModule)
import IfaceEnv (lookupOrigIO)
import OccName hiding (varName, mkTcOcc) -- (mkVarOcc, mkDataOcc)
import Data.Generics hiding (TyCon)
import Control.Monad (when, (<=<))
import TysWiredIn (consDataCon, consDataConName, nilDataConName, nilDataCon, intTy)
import PrelNames (metaConsDataConName)
import Name hiding (varName)
import TyCon (mkPrelTyConRepName, TyCon)
import Literal (LitNumType(..), Literal(..))
import CoreFVs (exprFreeVars)
import VarSet (VarSet, elemVarSet, emptyVarSet, unionVarSets, unitVarSet, mkVarSet, disjointVarSet)
import Debug.Trace (trace)
import HscTypes (lookupTyCon, lookupDataCon)
import Type (mkTyConApp, tyConAppTyCon, tyConAppArgs)
import FastString (fsLit)
import CoreUtils (exprType)
import CoreMonad (SimplMode)
import BasicTypes (CompilerPhase(..))
import Data.List (findIndex)

plugin :: Plugin
plugin = defaultPlugin 
  { installCoreToDos = install
  , pluginRecompile = purePlugin
  }

data State = State 
  { getRealRun :: Name
  , getRealRunId :: Id
  , getRun :: Name
  , getRunId :: Id
  , getPure :: DataCon
  , getImpure :: DataCon
  , getTree :: TyCon
  , getTreePrim :: TyCon
  , getBuffer :: TyCon
  , getError :: Id 
  , getCols :: Id
  , getRows :: Id
  , getSize :: TyCon
  , getUnary :: TyCon
  , getCrossRef :: DataCon
  , getReal :: DataCon
  , getFake :: DataCon
  }

matcherT :: (State -> TyCon) -> (State -> CoreExpr -> Bool)
matcherT f st (Type t)
  | tyConAppTyCon_maybe t == Just (f st) && null (tyConAppArgs t) = True
matcherT _ _ _ = False 

isBase :: State -> CoreExpr -> Bool
isBase st (App (App (Var e) t1) t2)
  | e == dataConWorkId (getPure st) && matcherT getTreePrim st t1 && matcherT getBuffer st t2 = True
{-isBase st (Var e)
  | e == getCols st = True
  | e == getRows st = True-}
isBase _ _ = False

isBase2 st e1 e2
 = e1 == dataConWorkId (getReal st) && (e2 == getCols st || e2 == getRows st)

within :: Var -> TyCon -> Bool
e `within` ty = e `elem` map dataConWorkId (visibleDataCons (algTyConRhs ty))

isSafe :: State -> CoreExpr -> Bool
isSafe st (Var e) 
  | e `within` getTreePrim st = True
  | e `within` getSize st = True
  | e `within` getUnary st = True
  | e == dataConWorkId (getImpure st) = True
isSafe st (App e1 e2) = isSafe st e1
isSafe _ _ = False

matchFirst v (RunExpr v1 _ _ _) = v == v1

resolveLen info v = (length info -) <$> findIndex (matchFirst v) info

substVars :: Info -> State -> CoreExpr -> CoreExpr
substVars info st e@(App a (Var b)) 
  | isBase st a = case resolveLen info b of
                    Nothing -> e
                    Just i -> App (Var (dataConWorkId (getCrossRef st))) (Lit (LitNumber LitNumInt (toInteger i) intTy))
substVars info st e@(App (Var e1) (App (Var e2) (Var e3)))
  | isBase2 st e1 e2 = case resolveLen info e3 of 
                         Nothing -> e
                         Just i -> App (Var (dataConWorkId (getFake st))) (Lit (LitNumber LitNumInt (toInteger i) intTy))
substVars info st (App a b)
  | isSafe st a = App (substVars info st a) (substVars info st b)
substVars _ _ e = e

getPrincipalVars :: State -> CoreExpr -> VarSet
getPrincipalVars st (App a (Var b)) 
  | isBase st a = unitVarSet b
getPrincipalVars st (App (Var e1) (App (Var e2) (Var e3)))
  | isBase2 st e1 e2 = unitVarSet e3
getPrincipalVars st (App a b)
  | isSafe st a = unionVarSets [getPrincipalVars st a, getPrincipalVars st b]
getPrincipalVars _ _ = emptyVarSet

getFreeVars :: State -> CoreExpr -> VarSet
getFreeVars st (App a (Var b)) 
  | isBase st a = emptyVarSet
getFreeVars st (App (Var e1) (App (Var e2) (Var e3)))
  | isBase2 st e1 e2 = emptyVarSet
getFreeVars st (App a b)
  | isSafe st a = unionVarSets [getFreeVars st a, getFreeVars st b]
getFreeVars _ e = exprFreeVars e

transformProgram :: ModGuts -> CoreM ModGuts
transformProgram guts = do
  hscEnv <- getHscEnv
  mod' <- liftIO $ findImportedModule hscEnv (mkModuleName "Lib") Nothing
  --mod'' <- liftIO $ findImportedModule hscEnv (mkModuleName "GHC.Types") Nothing
  let (Found _ mod) = mod'
  let getName = liftIO . lookupOrigIO hscEnv mod . mkVarOcc
  let getTyCon = lookupTyCon <=< (liftIO . lookupOrigIO hscEnv mod . mkTcOcc)
  let getDataCon = lookupDataCon <=< (liftIO . lookupOrigIO hscEnv mod . mkDataOcc)

  realRun <- getName "run"
  runInner <- getName "runInner"
  shouldNotBeReachable <- getName "shouldNotBeReachable"

  impure <- getDataCon "Impure"
  pur <- getDataCon "Pure"
  crossRef <- getDataCon "CrossRef"
  real <- getDataCon "Real"
  fake <- getDataCon "Fake"

  tree <- getTyCon "Tree"
  buffer <- getTyCon "Buffer"
  treePrim <- getTyCon "TreePrim"
  size <- getTyCon "SizePrim"
  unary <- getTyCon "UnaryOp"

  id <- lookupId shouldNotBeReachable
  runId <- lookupId runInner
  realRunId <- lookupId realRun
  cols <- lookupId =<< getName "cols"
  rows <- lookupId =<< getName "rows"

  let state = State
                { getRealRun = realRun
                , getRealRunId = realRunId
                , getRun = runInner
                , getRunId = runId
                , getTree = tree
                , getTreePrim = treePrim
                , getBuffer = buffer
                , getError = id
                , getImpure = impure
                , getPure = pur
                , getCols = cols
                , getRows = rows
                , getSize = size
                , getUnary = unary
                , getCrossRef = crossRef
                , getReal = real
                , getFake = fake
                }

  --putMsgS (showSDocUnsafe $ ppr name)
  newBinds <- mapM (transformFunc state guts) (mg_binds guts)
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

data RunExpr = RunExpr Var CoreExpr VarSet VarSet
type Info = [RunExpr]

present xs (RunExpr _ _ varSet _) = any (`elemVarSet` varSet) xs

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
getAll (RunExpr ns ls _ _:xs) = (ns:ns', ls:ls')
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

compat :: Info -> Info -> Bool
compat info1 info2 = provided `disjointVarSet` needed
  where provided = mkVarSet $ map (\(RunExpr a _ _ _) -> a) info1
        needed = unionVarSets $ map (\(RunExpr _ _ c _) -> c) info2

resolve :: Var -> CoreExpr -> (CoreExpr -> CoreExpr)
resolve v (Var v2) = substExpr (text "hashan") subst
  where subst = extendIdSubst emptySubst v (Var v2)
resolve _ _ = id

{-
names :: Info -> [(CoreExpr, [Name])]
names = mapM (\(RunExpr a b _) -> (a,) <$> )-}

transformBind :: State -> CoreBind -> CoreM CoreBind
transformBind st (NonRec b e) = (NonRec b) <$> (transformExpr' st e)
transformBind st (Rec bs) = Rec <$> (mapM (\(b, e) -> (b,) <$> transformExpr' st e) bs)

transformExpr' :: State -> CoreExpr -> CoreM CoreExpr
transformExpr' st e = do
  (e', info) <- transformExpr st e
  force st info e'

force :: State -> Info -> CoreExpr -> CoreM CoreExpr
force _ [] expr = return expr
force st [RunExpr x e _ _] expr = return $ Let (NonRec x (App (Var (getRealRunId st)) e)) expr
force st info expr = make st info expr

transformExpr :: State -> CoreExpr -> CoreM (CoreExpr, Info)

transformExpr st e@(App e1 e2)
  | matchesRun st e1 = do
        putMsgS "success"
        var2 <- mkSysLocalM (fsLit "hashan2") (getType'' st)
        putMsgS $ showSDocUnsafe (ppr (getPrincipalVars st e2))
        putMsgS $ showSDocUnsafe (ppr (getFreeVars st e2))
        --return (e, [])
        return (Var var2, [RunExpr var2 e2 (getFreeVars st e2) (getPrincipalVars st e2)])
  | otherwise = do
        (e1', info1) <- transformExpr st e1
        (e2', info2) <- transformExpr st e2
        let e' = App e1' e2'
        return (e', info1 ++ info2)

transformExpr st e@(Lam x e1) = do
  (e1', info) <- transformExpr st e1
  e1'' <- force st info e1'
  return $ (Lam x e1'', [])

transformExpr st e@(Let (NonRec x e1) e2) = do
  putMsgS "let"
  (e1', info1) <- transformExpr st e1
  (e2', info2) <- transformExpr st (resolve x e1' e2)
  if not (null (filter (present [x]) info2)) || not (info1 `compat` info2)
    then do
      e2'' <- force st info2 e2'
      return (Let (NonRec x e1') e2'', info1)
    else
      return (Let (NonRec x e1') e2', info1 ++ map (\(RunExpr a b c d) -> RunExpr a (substVars info1 st b) c d) info2)
  
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

install :: [CommandLineOption] -> [CoreToDo] -> CoreM [CoreToDo]
install _ todo = do
  putMsgS "Hello!"
  dflags <- getDynFlags
  let simplPass = SimplMode 
                    { sm_phase      = Phase 0
                    , sm_names      = []
                    , sm_dflags     = dflags
                    , sm_rules      = True
                    , sm_eta_expand = True
                    , sm_inline     = True
                    , sm_case_case  = True
                    }
  let pass = CoreDoPluginPass "Template" transformProgram
  return $ todo ++ [pass, CoreDoSimplify 3 simplPass]