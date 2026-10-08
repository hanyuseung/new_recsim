# Copyright 2019 The RecSim Authors. Licensed under the Apache License, Version 2.0.
"""Raw single- and multi-user simulators, independent of the Gymnasium adapter."""
import abc
from collections import OrderedDict
from copy import deepcopy
import numpy as np
from recsim import document
from recsim.config import validate_environment_config


class AbstractEnvironment(abc.ABC):
    def __init__(self, user_model, document_sampler, num_candidates, slate_size,
                 resample_documents=True):
        validate_environment_config(dict(num_candidates=num_candidates, slate_size=slate_size))
        self._user_model = user_model
        self._document_sampler = document_sampler
        self._seed = document_sampler._seed
        self._num_candidates = num_candidates
        self._slate_size = slate_size
        self._resample_documents = resample_documents
        if isinstance(user_model, list) and not user_model:
            raise ValueError('At least one user is required')
        self._ready = False
        self._do_resample_documents()

    @property
    def num_candidates(self):
        return self._num_candidates

    @property
    def slate_size(self):
        return self._slate_size

    @property
    def candidate_set(self):
        return self._candidate_set

    @property
    def user_model(self):
        return self._user_model

    @property
    def user_models(self):
        return self.user_model if isinstance(self, MultiUserEnvironment) else [self.user_model]

    def _do_resample_documents(self):
        self._candidate_set = document.CandidateSet()
        for _ in range(self.num_candidates):
            self._candidate_set.add_document(self._document_sampler.sample_document())
        if self._candidate_set.size() != self.num_candidates:
            raise ValueError('Document sampler produced duplicate candidate IDs')
        self._current_documents = OrderedDict(self._candidate_set.create_observation())

    def reset_sampler(self, seed=None):
        if seed is not None:
            self._seed = seed
        seed = getattr(self, '_seed', self._document_sampler._seed)
        streams = np.random.SeedSequence(seed).spawn(1 + len(self.user_models))
        self._document_sampler.reset_sampler(int(streams[0].generate_state(1)[0]))
        for model, stream in zip(self.user_models, streams[1:]):
            model.reset_sampler(int(stream.generate_state(1)[0]))
        # Even a fixed candidate pool must replay when explicitly reseeded.
        self._do_resample_documents()
        self._ready = False

    def reset(self):
        for model in self.user_models:
            model.reset()
        if self._resample_documents:
            self._do_resample_documents()
        self._ready = True
        users = [deepcopy(m.create_observation()) for m in self.user_models]
        return (users if isinstance(self, MultiUserEnvironment) else users[0],
                deepcopy(self._current_documents))

    def _validate_slate(self, slate):
        action = np.asarray(slate)
        if action.shape != (self.slate_size,) or action.dtype.kind not in 'iu':
            raise ValueError(f'Expected {self.slate_size} integer candidate indices')
        if np.any(action < 0) or np.any(action >= self.num_candidates):
            raise ValueError('Candidate index is out of range')
        return action

    def _step(self, slates):
        if not self._ready:
            raise RuntimeError('Call reset() before step() or after episode termination')
        if len(slates) != len(self.user_models):
            raise ValueError('Expected one slate per user')
        # Validate every action before mutating any user.
        slates = [self._validate_slate(s) for s in slates]
        ids = list(self._current_documents)
        all_responses, active_docs, active_responses = [], [], []
        for model, slate in zip(self.user_models, slates):
            docs = self.candidate_set.get_documents([ids[i] for i in slate])
            if model.is_terminal():
                responses = []
            else:
                responses = model.simulate_response(docs)
                if len(responses) != len(docs):
                    raise ValueError('A user model must return one response per document')
                model.update_state(docs, responses)
                active_docs.extend(docs)
                active_responses.extend(responses)
            all_responses.append(responses)
        self._document_sampler.update_state(active_docs, active_responses)
        users = [deepcopy(m.create_observation()) for m in self.user_models]
        done = all(m.is_terminal() for m in self.user_models)
        self._ready = not done
        if self._resample_documents:
            self._do_resample_documents()
        return users, deepcopy(self._current_documents), all_responses, bool(done)

    @abc.abstractmethod
    def step(self, slate):
        """Return raw user/doc/response/done values."""


class SingleUserEnvironment(AbstractEnvironment):
    def step(self, slate):
        users, docs, responses, done = self._step([slate])
        return users[0], docs, responses[0], done


Environment = SingleUserEnvironment


class MultiUserEnvironment(AbstractEnvironment):
    @property
    def num_users(self):
        return len(self.user_model)

    def step(self, slates):
        return self._step(slates)
