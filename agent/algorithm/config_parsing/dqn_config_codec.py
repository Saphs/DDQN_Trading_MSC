import json
from json import JSONEncoder
from pathlib import Path
from typing import Any, Callable

from agent.algorithm.config_parsing.agent_parameters import AgentParameters
from agent.algorithm.config_parsing.dqn_config import DqnConfig
from agent.algorithm.config_parsing.environment_parameters import EnvironmentParameters


class DqnConfigCodec(JSONEncoder):
    """Config codec doing the heavy lifting for the DqnConfig class when it comes to serialization."""

    type_hint = '__type__'

    @classmethod
    def read_json(cls, p: Path) -> "DqnConfig":
        with open(p, 'r') as f:
            j_str = ''.join(f.readlines())
        return json.loads(j_str, object_hook=DqnConfigCodec.decoder_hook())

    @classmethod
    def to_json(cls, c: DqnConfig, p: Path) -> None:
        with open(p, 'w') as f:
            f.write(json.dumps(c, cls=DqnConfigCodec, indent=2))

    @classmethod
    def decoder_hook(cls) -> Callable[[dict], Any]:
        """Return function usable as a decoder without any class signature."""
        def _f(d: dict) -> Any:
            if cls.type_hint not in d:
                return d
            else:
                if d[cls.type_hint] == DqnConfig.__name__:
                    del d[cls.type_hint]
                    return DqnConfig(**d)
                elif d[cls.type_hint] == AgentParameters.__name__:
                    del d[cls.type_hint]
                    return AgentParameters(**d)
                elif d[cls.type_hint] == EnvironmentParameters.__name__:
                    del d[cls.type_hint]
                    return EnvironmentParameters(**d)
                return d
        return _f

    def default(self, o: Any) -> Any:
        d: dict = o.__dict__
        d[self.type_hint] = o.__class__.__name__
        return d
