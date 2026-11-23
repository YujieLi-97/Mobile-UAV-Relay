
import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import uav_relay_experiment
uav_relay_experiment._install_modules()
import numpy as np
from unittest.mock import patch
from sim import Cfg,rx_power_dbm,i_max_dbm,e2e_rate,TrueField,make_ground_track
from planner import GaussianField,MPPIPlanner
from graph_search_baseline import LatticePlanner,ExactOraclePlanner
import experiment as E
class ScienceTests(unittest.TestCase):
 def test_boundary_safe_projection(self):
  position=np.array([1990.,1990.]); previous=np.zeros(2)
  rng=np.random.default_rng(71)
  for _ in range(500):
   velocity=uav_relay_experiment._safe_velocity(position,previous,rng.normal(0,20,2),Cfg)
   self.assertLessEqual(np.linalg.norm(velocity-previous),Cfg.A_MAX*Cfg.DT+1e-6)
   self.assertLessEqual(np.linalg.norm(velocity),Cfg.V_MAX+1e-6)
   position=position+velocity*Cfg.DT
   self.assertTrue(np.all(position>=-1e-6) and np.all(position<=Cfg.L+1e-6))
   previous=velocity
 def test_link_budget(self):
  q=np.array([1250.,1250.]);g=np.array([1500.,1500.])
  d=np.sqrt(250**2*2+100**2);p=rx_power_dbm(d)
  self.assertAlmostEqual(d,367.42346,places=4)
  self.assertAlmostEqual(float(p),-66.43,delta=.01)


  self.assertAlmostEqual(float(i_max_dbm(np.array([p]))[0]),-94.624,delta=.002)
  self.assertAlmostEqual(float(e2e_rate(q,g,-95.,-88.))/1e6,3.544,delta=.002)
  self.assertLess(float(e2e_rate(q,g,-95.,-80.)),float(e2e_rate(q,g,-95.,-88.)))
 def test_gp_recent_observations_and_variance(self):
  gp=GaussianField(max_obs=20)
  for _ in range(20):gp.add([500.,500.,100.],-90.)
  mu,sd=gp.predict([[500.,500.,100.],[1500.,1500.,100.]])
  self.assertLess(sd[0],.23);self.assertGreater(sd[1],11.)
  for i in range(30):gp.add([i,0.,100.],-90.)
  np.testing.assert_array_equal(gp.X[:,0],np.arange(10,30))
 def test_causal_velocity_and_frozen_baseline(self):
  cfg=type('Short',(Cfg,),{'T_STEPS':8})
  velocities=[]
  def step(pl,q,gp,g,v):velocities.append(v.copy());return np.zeros(2)
  with patch.object(MPPIPlanner,'step',step):
   r=E.run_episode(12,'twophase',te_steps=2,cfg=cfg,record=True,grid_reference=11)
  self.assertLessEqual(len(r['_gp'].y),4)
  gt=r['_gtraj']
  for k,v in enumerate(velocities,2):np.testing.assert_allclose(v,(gt[k]-gt[k-1])/cfg.DT)
  velocities.clear()
  with patch.object(MPPIPlanner,'step',step):E.run_episode(12,'ours',cfg=cfg,grid_reference=11)
  np.testing.assert_array_equal(velocities[0],[0.,0.])
 def test_reproducibility(self):
  cfg=type('Short',(Cfg,),{'T_STEPS':12})
  a=E.run_episode(91,'ours',lam=2.,cfg=cfg,record=True,grid_reference=11)
  b=E.run_episode(91,'ours',lam=2.,cfg=cfg,record=True,grid_reference=11)
  np.testing.assert_array_equal(a['_rates'],b['_rates']);np.testing.assert_array_equal(a['_traj'],b['_traj'])
 def test_speed_acceleration(self):
  p=MPPIPlanner(a_max=4.,seed=5);gp=GaussianField();q=np.array([1000.,1000.]);v0=np.zeros(2)
  for _ in range(10):
   v=p.step(q,gp,np.array([1800.,1000.]),np.array([5.,0.]))
   self.assertLessEqual(np.linalg.norm(v),20.+1e-8)
   self.assertLessEqual(np.linalg.norm(v-v0),4.+1e-8);q+=v;v0=v
 def test_static_fading_enabled(self):
  cfg=type('Short',(Cfg,),{'T_STEPS':12})
  a=E.run_episode(6,'static',cfg=cfg,record=True,grid_reference=11,use_fading=False)
  b=E.run_episode(6,'static',cfg=cfg,record=True,grid_reference=11,use_fading=True)
  self.assertFalse(np.array_equal(a['_rates'],b['_rates']))
 def test_lattice_actions_obey_motion_limit(self):
  planner=LatticePlanner(depth=4)
  gp=GaussianField()
  q=np.array([1000.,1000.])
  for _ in range(8):
   velocity=planner.step(q,gp,np.array([1500.,1200.]),np.array([2.,1.]))
   self.assertLessEqual(np.linalg.norm(velocity),Cfg.V_MAX+1e-8)
   self.assertLessEqual(np.linalg.norm(velocity-planner.last_velocity),Cfg.A_MAX*Cfg.DT+1e-8)
   q=np.clip(q+velocity*Cfg.DT,0.,Cfg.L)
   planner.last_velocity=velocity
 def test_hindsight_track_replays_scene(self):
  seed=81
  rng=np.random.default_rng(seed)
  field=TrueField(rng)
  expected=make_ground_track(rng)
  planner=ExactOraclePlanner(seed=seed,oracle_fn=field)
  np.testing.assert_array_equal(planner.user_track,expected)
 def test_jammer_altitude_and_executed_acceleration(self):
  f=TrueField(np.random.default_rng(13))
  ground=float(f(np.r_[f.p_J,0.])[0])
  air=float(f(np.r_[f.p_J,Cfg.H_UAV])[0])
  self.assertGreater(ground-air,50.)
  cfg=type('Short',(Cfg,),{'T_STEPS':20})
  with patch.object(MPPIPlanner,'step',lambda pl,q,gp,g,v:np.array([20.,0.]) if len(gp.y)%2 else np.array([-20.,0.])):
   result=E.run_episode(14,'ours',cfg=cfg,record=True,grid_reference=11)
  velocities=np.diff(np.vstack([result['_traj'][0],result['_traj']]),axis=0)/cfg.DT
  self.assertTrue(np.all(np.linalg.norm(np.diff(velocities,axis=0),axis=1)<=cfg.A_MAX*cfg.DT+1e-8))
if __name__=='__main__':unittest.main(verbosity=2)
