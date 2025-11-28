# file: env_wrappers.py (最终完整修正版)

import multiprocessing as mp
import numpy as np

# 'worker' 和 'CloudpickleWrapper' 保持不变
def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            # 【最终修正】移除在子进程中的自动重置逻辑
            # if 'bool' in done.__class__.__name__:
            #     if done:
            #         ob = env.reset()
            # else:
            #     if np.all(done):
            #         ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.share_observation_space, env.action_space))
        else:
            raise NotImplementedError

class CloudpickleWrapper(object):
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

# 'SubprocVecEnv' 保持不变
class SubprocVecEnv():
    def __init__(self, env_fns, spaces=None):
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(nenvs)])
        self.ps = [mp.Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(('get_spaces', None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv()
        self.observation_space = observation_space
        self.share_observation_space = share_observation_space
        self.action_space = action_space
    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()
    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True
    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos
    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])
    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

# 单环境
class DummyVecEnv():
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        self.num_envs = len(env_fns)
        self.observation_space = env.observation_space
        self.share_observation_space = env.share_observation_space
        self.action_space = env.action_space
        self.actions = None

    def get_f_g_k(self):
        if not self.envs:
            return 1 
        return self.envs[0].get_f_g_k()
    
    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, rews, dones, infos = map(np.array, zip(*results))

        self.actions = None
        return obs, rews, dones, infos

    def reset(self):
        obs = [env.reset() for env in self.envs]
        return np.array(obs)

    def close(self):
        for env in self.envs:
            env.close()

    def render(self, mode="human"):
        if mode == "rgb_array":
            return np.array([env.render(mode=mode) for env in self.envs])
        elif mode == "human":
            for env in self.envs:
                env.render(mode=mode)
        else:
            raise NotImplementedError