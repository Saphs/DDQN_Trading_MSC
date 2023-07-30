import json
import os
import unittest
from pathlib import Path

from agent.algorithm.config_parsing.agent_parameters import AgentParameters
from agent.algorithm.config_parsing.dqn_config import DqnConfig
from agent.algorithm.config_parsing.dqn_config_codec import DqnConfigCodec
from agent.algorithm.config_parsing.environment_parameters import EnvironmentParameters


class TestDqnConfig(unittest.TestCase):

    default_constructor_dqn_config_serialized: Path = "./resources/default_constructor_dqn_config_serialized.json"
    target_folder: Path = Path("./resources")

    def test_default_constructor(self):
        """Check if default construction build valid DqnConfig with the correct types."""
        config: DqnConfig = DqnConfig()
        self.assertEqual(type(config), DqnConfig)
        self.assertEqual(type(config.agent), AgentParameters)
        self.assertEqual(type(config.environment), EnvironmentParameters)
        self.assertIsNotNone(config)

    def test_default_config_serializes_and_deserializes(self):
        """Check if default DqnConfig can be serialized and deserialized."""
        config: DqnConfig = DqnConfig()

        # Confirm expected serialization
        serialized: str = json.dumps(config, cls=DqnConfigCodec, indent=2)
        with open(self.default_constructor_dqn_config_serialized, 'r') as f:
            expected_json_str = ''.join(f.readlines())
        self.assertEqual(serialized, expected_json_str)

        # Confirm expected deserialization
        deserialized: DqnConfig = json.loads(serialized, object_hook=DqnConfigCodec.decoder_hook())
        self.assertEqual(deserialized, config)
        self.assertEqual(type(deserialized), DqnConfig)
        self.assertEqual(type(deserialized.agent), AgentParameters)
        self.assertEqual(type(deserialized.environment), EnvironmentParameters)

    def test_read_and_write_json(self):
        """Check if DqnConfig can be saved and loaded to disk."""
        config: DqnConfig = DqnConfig()
        temp_file = Path.joinpath(self.target_folder, 'temp_config.json')
        DqnConfigCodec.to_json(config, temp_file)
        loaded_config: DqnConfig = DqnConfigCodec.read_json(temp_file)
        os.remove(temp_file)

        self.assertEqual(config, loaded_config)
        self.assertEqual(type(loaded_config), DqnConfig)
        self.assertEqual(type(loaded_config.agent), AgentParameters)
        self.assertEqual(type(loaded_config.environment), EnvironmentParameters)




