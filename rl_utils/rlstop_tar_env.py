import numpy as np


import gymnasium as gym
from gymnasium import spaces


class TAREnv(gym.Env):



    def __init__(self, target_recall, topics_list = None, topic_id= None, size=100 , render_mode=None):
        self.size = size  # The size of the ranking relv vector

        #observation is 1D np array size array of relv vector
        self.observation_space = spaces.Box(-1,  1, shape=(size,), dtype=np.float32)

        #  actions, corresponding to "next", "stop"
        self.action_space = spaces.Discrete(2)

        # Set up properties
        self.done = False
        self.reward = 0
        self.hit, self.miss = 0, 0


        # Set up the TAR
        self.vector_size = size


        # current position and stop position
        self._agent_location = 0
        self._target_location = -1 #dummy value

        # keep predicted recall so far
        self.recall = 0
        self.target_recall = target_recall

        # topic data
        self.topics_list = topics_list
        self.topic_id = topic_id # for single env


        #for vec env
        if topic_id is None:
          # select a new random topic
          while True:
            t = random.choice(topics_list)
            if t not in SELECTED_TOPICS:
              SELECTED_TOPICS.append(t)
              self.topic_id = t
              break

          # use same ordered list of topics across diffreent runs
          if TRAINING:
            global SELECTED_TOPICS_ORDERERD_INDEX
            self.topic_id = SELECTED_TOPICS_ORDERERD[SELECTED_TOPICS_ORDERERD_INDEX]
            SELECTED_TOPICS_ORDERERD_INDEX += 1
        else:
           self.topic_id = topic_id # for single env


        self.n_docs = 0
        self.rel_cnt = []
        self.rel_rate = []
        self.n_samp_docs = 0
        self.n_samp_docs_after_target = 0
        self.n_samp_docs_current = []
        self.rel_list = []
        self.all_vectors = []

        # Define constants for clear code
        self.NEXT = 0
        self.STOP = 1


        self.load_data_flag = True
        self.load_data(self.topic_id)

        self.first_step_flag = True




    def load_data(self, topic_id):

      all_vectors = [[-1]*self.vector_size for i in range(self.vector_size)]
      topic_id = self.topic_id

      n_docs = len(doc_rank_dic[topic_id])  # total n. docs in topic
      rel_list = rank_rel_dic[topic_id]  # list binary rel of ranked docs

      # get points
      windows = make_windows(self.vector_size, n_docs)

      window_size = windows[0][1]

      # calculate relv vector points (batches)
      rel_cnt,rel_rate, n_samp_docs_current = get_rel_cnt_rate(windows, self.vector_size, rel_list)

      self.n_docs = n_docs
      self.rel_cnt = rel_cnt
      self.rel_rate = rel_rate
      self.n_samp_docs_current = n_samp_docs_current
      self.rel_list = rel_list


      #update all vector with all possible examined states
      for i in range(self.vector_size):
        all_vectors[i][0:i+1] = rel_rate[0:i+1] # update examined part

        #calculate target recall stopping pos
        #mark only 1st recall achieved stopping position
        if (sum(self.rel_cnt[0:i+1]) / sum(self.rel_cnt)) >= self.target_recall and self._target_location == -1:
          self._target_location = i


      self.all_vectors = all_vectors


    def _get_obs(self):
        return  np.array(self.all_vectors[self._agent_location], dtype=np.float32)


    def _get_info(self):
        return {
                "topic_id": self.topic_id,
                "recall": round((self.recall),3),
                "cost": round(((self._agent_location +1)/100),3), # each vec pos == 1% of collection
                "e_cost": round((((self._agent_location +1) - self._target_location)/100) / (1-(self._target_location/100)),3), #updated
                "distance": (self._agent_location - self._target_location),
                "agent": (self._agent_location),
                "target": (self._target_location),
                "agent_vector": np.array(self.all_vectors[self._agent_location]),
                "terminal_observation": np.array(self.all_vectors[self._target_location])} # target_vector named terminal_observation needed for SB3 vec_env

    def reset(self,seed=0):

        # re-load data 1st time for vec_env
        if self.load_data_flag:
          self.load_data(self.topic_id)
          self.load_data_flag = False

        #initialize all pos info in reset (i.e. recall, n_samp_docs)
        # always start at first position of relv vector (1st batch size)
        self._agent_location = 0
        self.n_samp_docs =  sum(self.n_samp_docs_current[0:self._agent_location+1])
        self.n_samp_docs_after_target =  sum(self.n_samp_docs_current[self._target_location:self._agent_location+1])

        self.recall = sum(self.rel_cnt[0:self._agent_location+1]) / sum(self.rel_cnt)

        state = self.all_vectors[self._agent_location]


        observation = self._get_obs()
        info = self._get_info()

        return observation, info


    def step(self, action):
        truncated = False
        terminated = False

        if self._agent_location >= self.vector_size-1:
          self.done = True
          truncated = True


        if self._agent_location >= self.vector_size-2 and action == self.NEXT:
          self.done = True
          truncated = True

        if action == self.STOP:
            terminated = True


        if action == self.NEXT:
            #update TAR vars to next pos vars
            if self.first_step_flag:
              self._agent_location = self._agent_location # dont move next, examine 1st portion at pos [0]
              self.first_step_flag = False
            else:
              self._agent_location += 1 # move to next portion (examined)

            self.n_samp_docs =  sum(self.n_samp_docs_current[0:self._agent_location+1])
            self.n_samp_docs_after_target =  sum(self.n_samp_docs_current[self._target_location:self._agent_location+1])
            self.recall = sum(self.rel_cnt[0:self._agent_location+1]) / sum(self.rel_cnt)

        observation = self._get_obs()
        info = self._get_info()


        # R(t) = 1 - (t-1) / target_step, if t ≤ target_step , -1 * (t - target_step) / (total_steps - target_step), if t > target_step

        #to get more easy readable formula
        reward_target_location = self._target_location+1
        reward_agent_location = self._agent_location+1

        # misses/reach target_recall
        if reward_agent_location <= reward_target_location:
          self.reward = 1 - ((reward_agent_location-1) / reward_target_location)

        # overachieves target_recall
        elif reward_agent_location > reward_target_location:
            self.reward = -1 * (reward_agent_location - reward_target_location) / (self.vector_size - reward_target_location)

        return observation, self.reward, terminated, truncated, info




    def render(self):
        # we dont need render
        return


    def close(self):
        # we dont need close
        return

