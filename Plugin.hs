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
import TysWiredIn (consDataCon, consDataConName, nilDataConName, nilDataCon, intDataCon, trueDataCon, falseDataCon, boolTy)
import TysPrim (intPrimTy)
import PrelNames (metaConsDataConName)
import Name hiding (varName)
import TyCon (mkPrelTyConRepName, TyCon)
import Literal (LitNumType(..), Literal(..), nullAddrLit)
import CoreFVs (exprFreeVars)
import VarSet (VarSet, elemVarSet, emptyVarSet, unionVarSets, unitVarSet, mkVarSet, disjointVarSet)
import Debug.Trace (trace)
import HscTypes (lookupTyCon, lookupDataCon)
import Type (mkTyConApp, tyConAppTyCon, tyConAppArgs)
import FastString (fsLit)
import CoreUtils (exprType)
import CoreMonad (SimplMode)
import BasicTypes (CompilerPhase(..))
import Data.List (findIndex, isPrefixOf)
import Simplify (simplExpr)
import SimplEnv (mkSimplEnv, SimplEnv, seDynFlags)
import SimplMonad (initSmpl, SimplM)
import FamInstEnv (emptyFamInstEnv)
import CoreSyn (emptyRuleEnv)
import CoreStats (exprSize)
import OccurAnal (occurAnalyseExpr)

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
  , getFakeRows :: DataCon
  , getFakeCols :: DataCon
  , getFree :: TyCon
  , getSimplEnv :: SimplEnv
  , getIndent :: Int
  }

putIndent :: State -> String -> CoreM ()
putIndent st s = putMsgS $ (take (getIndent st) (repeat ' ')) ++ s

incr :: State -> State
incr st = st { getIndent = (getIndent st + 2) }

runSimpl :: State -> Int -> SimplM a -> CoreM a
runSimpl st i comp = do
  let dynFlags = seDynFlags $ getSimplEnv st
  let ruleEnv = emptyRuleEnv
  let fam = (emptyFamInstEnv, emptyFamInstEnv)
  uniqSupply <- getUniqueSupplyM
  (a, _) <- liftIO $ initSmpl dynFlags ruleEnv fam uniqSupply i comp
  return a

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

num i = App (Var (dataConWorkId intDataCon)) (Lit (LitNumber LitNumInt (toInteger i) intPrimTy))

getAppropriate :: State -> Var -> Var
getAppropriate st v
  | v == getCols st = dataConWorkId $ getFakeCols st
  | v == getRows st = dataConWorkId $ getFakeRows st

substVars :: Info -> State -> CoreExpr -> CoreExpr
substVars info st e@(App a (Var b)) 
  | isBase st a = case resolveLen info b of
                    Nothing -> e
                    Just i -> App impureApped (App (App crossRef (Type freeTy)) (num i))
  where bufferTy = mkTyConApp (getBuffer st) []
        treePrimTy = mkTyConApp (getTreePrim st) []
        freeTy = mkTyConApp (getFree st) [treePrimTy, bufferTy]
        crossRef = Var (dataConWorkId (getCrossRef st))
        impure = Var (dataConWorkId (getImpure st))
        impureApped = App (App impure (Type treePrimTy)) (Type bufferTy)
substVars info st e@(App (Var e1) (App (Var e2) (Var e3)))
  | isBase2 st e1 e2 = case resolveLen info e3 of 
                         Nothing -> e
                         Just i -> App (Var (getAppropriate st e2)) (num i)
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

transformProgram :: SimplEnv -> ModGuts -> CoreM ModGuts
transformProgram env guts = do
  hscEnv <- getHscEnv
  mod' <- liftIO $ findImportedModule hscEnv (mkModuleName "Lib") Nothing
  otherMod' <- liftIO $ findImportedModule hscEnv (mkModuleName "Control.Exception.Base") Nothing
  --mod'' <- liftIO $ findImportedModule hscEnv (mkModuleName "GHC.Types") Nothing
  let (Found _ mod) = mod'
  let (Found _ otherMod) = otherMod'
  let getName = liftIO . lookupOrigIO hscEnv mod . mkVarOcc
  let getTyCon = lookupTyCon <=< (liftIO . lookupOrigIO hscEnv mod . mkTcOcc)
  let getDataCon = lookupDataCon <=< (liftIO . lookupOrigIO hscEnv mod . mkDataOcc)

  realRun <- getName "run"
  runInner <- getName "runInner"
  -- shouldNotBeReachable <- getName "shouldNotBeReachable"
  shouldNotBeReachable <- liftIO (lookupOrigIO hscEnv otherMod  (mkVarOcc "patError"))

  impure <- getDataCon "Impure"
  pur <- getDataCon "Pure"
  crossRef <- getDataCon "CrossRef"
  real <- getDataCon "Real"
  fakeRows <- getDataCon "FakeRows"
  fakeCols <- getDataCon "FakeCols"

  tree <- getTyCon "Tree"
  buffer <- getTyCon "Buffer"
  treePrim <- getTyCon "TreePrim"
  size <- getTyCon "SizePrim"
  unary <- getTyCon "UnaryOp"
  free <- getTyCon "Free"

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
                , getFakeRows = fakeRows
                , getFakeCols = fakeCols
                , getFree = free
                , getSimplEnv = env
                , getIndent = 0
                }

  --putMsgS (showSDocUnsafe $ ppr name)
  newBinds <- mapM (transformFunc state guts) (mg_binds guts)
  return $ guts { mg_binds = newBinds }

transformFunc :: State -> ModGuts -> CoreBind -> CoreM CoreBind
transformFunc st guts x = do
  putMsgS "---"
  let b = "doBatch" `isPrefixOf` (getOccString $ head (getNames x))
  -- b <- shouldTransformBind guts x
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

present xs (RunExpr _ _ varSet1 varSet2) = any (`elemVarSet` varSet1) xs || any (`elemVarSet` varSet2) xs

getType :: State -> Type
getType st = mkTyConApp (getTree st) [mkTyConApp (getBuffer st) []]

getType' :: State -> Type
getType' st = mkTyConApp listTyCon [mkTyConApp (getBuffer st) []]

getType'' :: State -> Type
getType'' st = mkTyConApp (getBuffer st) []

createRun :: State -> Type -> [CoreExpr] -> CoreM CoreExpr
createRun st ty [] = do
  return $ App (Var (dataConWorkId nilDataCon)) (Type ty)
createRun st ty (x:xs) = do
  rest <- createRun st ty xs
  return $ App (App (App (Var (dataConWorkId consDataCon)) (Type ty)) x) rest

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
  let t = tyConAppArgs (typeKind ty) !! 0
  return $ Case e var1 ty 
    [(DataAlt nilDataCon, [], App (App (App (Var (getError st)) (Type t)) (Type ty)) (Lit nullAddrLit)),
     (DataAlt consDataCon, [var2, var3], rest)]

make :: State -> Info -> CoreExpr -> CoreM CoreExpr
make st info e = do
  let freeVars = exprFreeVars e
  let (ns, ls) = getAll info
  let present = map (\a -> if a `elemVarSet` freeVars 
                  then Var (dataConWorkId trueDataCon) 
                  else Var (dataConWorkId falseDataCon)) ns
  run1 <- createRun st (getType st) ls
  run2 <- createRun st boolTy present
  let ns' = filter (`elemVarSet` freeVars) ns
  let expr = App (App (Var (getRunId st)) run1) run2
  createDown st (exprType e) ns' expr e

compat :: Info -> Info -> Bool
compat info1 info2 = provided `disjointVarSet` needed
  where provided = mkVarSet $ map (\(RunExpr a _ _ _) -> a) info1
        needed = unionVarSets $ concatMap (\(RunExpr _ _ c d) -> [c{-,d-}]) info2

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
        --putMsgS "success"
        var2 <- mkSysLocalM (fsLit "hashan2") (getType'' st)
        --putMsgS $ showSDocUnsafe (ppr (getPrincipalVars st e2))
        --putMsgS $ showSDocUnsafe (ppr (getFreeVars st e2))
        --return (e, [])
        return (Var var2, [RunExpr var2 e2 (getFreeVars st e2) (getPrincipalVars st e2)])
  | otherwise = do
        (e1', info1) <- transformExpr st e1
        (e2', info2) <- transformExpr st e2
        let e' = App e1' e2'
        return (e', info1 ++ info2)

transformExpr st e@(Lam x e1) = do
  putIndent st $ "lam " ++ showSDocUnsafe (ppr x) ++ " {"
  (e1', info) <- transformExpr (incr st) e1
  putIndent st $ "}"
  if not (null (filter (present [x]) info))
    then do
      e1'' <- force st info e1'
      return $ (Lam x e1'', [])
    else
      return $ (Lam x e1', info)

transformExpr st e@(Let (NonRec x e1) e2) = do
  putIndent st $ "lethead " ++ showSDocUnsafe (ppr x) ++ " {"
  (e1', info1) <- transformExpr (incr st) e1
  if null info1 
    then do
      putIndent st $ "} letbase {" -- ++ showSDocUnsafe (ppr x)
      (e2', info2) <- transformExpr (incr st) e2
      e2'' <- force st info2 e2'
      putIndent st $ "}"
      return (Let (NonRec x e1') e2'', [])
    else do
      putIndent st $ "} letrec {" -- ++ showSDocUnsafe (ppr x) ++ " " ++ (show (length info1))
      let expToSimplify = Let (NonRec x e1') e2
      e' <- occurAnalyseExpr <$> (runSimpl st (exprSize expToSimplify) $ simplExpr (getSimplEnv st) expToSimplify)
      (e'', info2) <- transformExpr (incr st) e'
      if info1 `compat` info2
        then do
          let info' = foldl (\acc (RunExpr a b c d) -> acc ++ [RunExpr a (substVars acc st b) c d]) info1 info2
          putIndent st $ "} " ++ (show (length info'))
          return (e'', info')
        else do
          e''' <- force st info2 e''
          putIndent st $ "} " ++ (show (length info1))
          return (e''', info1)

  {-(e2', info2) <- transformExpr st (resolve x e1' e2)
  --let info2' = map (\(RunExpr a b c d) -> RunExpr a (substVars info1 st b) c d) info2
  if not (null (filter (present [x]) info2)) || not (info1 `compat` info2)
    then do
      e2'' <- force st info2 e2'
      return (Let (NonRec x e1') e2'', info1)
    else
      let p = exprFreeVars e2' in
      let info' = foldl (\acc (RunExpr a b c d) -> acc ++ [RunExpr a (substVars acc st b) c d]) info1 info2 in
      if x `elemVarSet` p
        then return (Let (NonRec x e1') e2', info') 
        else return (e2', info')-}

transformExpr st (Case e1 x t [(f, g, e2)]) = do
  putIndent st $ "caseOne " ++ showSDocUnsafe (ppr x) ++ " {"
  (e1', info1) <- transformExpr (incr st) e1
  (e2', info2) <- transformExpr (incr st) (resolve x e1' e2)
  putIndent st "}"
  if not (null (filter (present (x:g)) info2)) || not (info1 `compat` info2)
    then do
      e2'' <- force st info2 e2'
      return (Case e1' x t [(f, g, e2'')], info1)
    else
      let p = exprFreeVars e2' in
      let info' = foldl (\acc (RunExpr a b c d) -> acc ++ [RunExpr a (substVars acc st b) c d]) info1 info2 in
      if any (`elemVarSet` p) (x:g)
        then return (Case e1' x t [(f, g, e2')], info')
        else return (e2', info')

transformExpr st (Case e1 b t as) = do
  putIndent st $ "case " ++ showSDocUnsafe (ppr b) ++ " {"
  (e1', info1) <- transformExpr (incr st) e1
  as' <- mapM (\(a, b, c) -> ((a,b,) <$> transformExpr' (incr st) c)) as
  putIndent st "}"
  return (Case e1' b t as', info1)

transformExpr st e@(Type ty) = return (e, [])

transformExpr st e@(Coercion co) = return (e, [])

transformExpr st (Cast e co) = do
  (e', info) <- transformExpr (incr st) e
  return (Cast e' co, info)

transformExpr st e@(Var id) = do
  putIndent st $ "var " ++ (showSDocUnsafe (ppr id))
  return (e, [])

transformExpr st e = do
  putIndent st $ "bailout"
  return (e, [])

--let (e1', info1) = (e1, [])
{-transformAlts st = do
  (g', info) <- transformExpr st g
  if not (null (filter (present h) info))
    then do
      g'' <- force st info g'
      return ([(f, h, g'')], [])
    else
      return ([(f, h, g')], info)-}

install :: [CommandLineOption] -> [CoreToDo] -> CoreM [CoreToDo]
install _ todo = do
  putMsgS "Hello!"
  dflags <- getDynFlags
  {-let simplPass = SimplMode 
                    { sm_phase      = Phase 0
                    , sm_names      = []
                    , sm_dflags     = dflags
                    , sm_rules      = True
                    , sm_eta_expand = True
                    , sm_inline     = True
                    , sm_case_case  = True
                    }-}
  let simplPass = SimplMode 
                    { sm_phase      = Phase 0
                    , sm_names      = []
                    , sm_dflags     = dflags
                    , sm_rules      = True
                    , sm_eta_expand = False
                    , sm_inline     = True
                    , sm_case_case  = False
                    }
  let simplEnv = mkSimplEnv simplPass
  let pass = CoreDoPluginPass "Template" $ transformProgram simplEnv
  return $ todo ++ [pass {- CoreDoSimplify 3 simplPass -}]