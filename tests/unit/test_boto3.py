from gofannon.open_notify_space.iss_locator import IssLocator
import json
import unittest
from unittest.mock import patch

perfect_agent_app_config = {
    "app_id": "gofannon_demo",
    "agent_name": "ISSLocator",
    "agent_session_timeout": 600,
    "instruction": "You are an agent skilled at tracking and reporting the location of the International Space Station. Your role is to help users understand where the ISS is currently located by providing its geographic coordinates and other relevant location details.",
    "agent_description": "This agent helps to locate the location of the ISS.",
    "target_model": "anthropic.claude-3-sonnet-20240229-v1:0",
    "python_runtime_version": "python3.13",
    "temp_build_root": "/tmp/build",
}


class TestFailureModule(unittest.TestCase):

    def test_boto3_invalid_agent_app_config(self):
        bogus_agent_app_config = perfect_agent_app_config
        del bogus_agent_app_config["app_id"]
        iss_locator = IssLocator()
        with self.assertRaises(RuntimeError) as context:
            bedrock_config = iss_locator.export_to_bedrock(
                agent_app_config=bogus_agent_app_config
            )
        expected_error = "JSON validation failure on input parameters: 'app_id' is a required property"
        self.assertEqual(expected_error, str(context.exception)[: len(expected_error)])

    def test_boto3_no_valid_aws_account_number(self, MockDependency):

        mock_instance = MockDependency.return_value
        mock_instance.method_to_stub.return_value = 10

        iss_locator = IssLocator()
        with self.assertRaises(RuntimeError) as context:
            bedrock_config = iss_locator.export_to_bedrock(
                agent_app_config=perfect_agent_app_config
            )
        # expected_error = "JSON validation failure on input parameters: 'app_id' is a required property"
        # self.assertEqual(expected_error, str(context.exception)[: len(expected_error)])


if __name__ == "__main__":
    unittest.main()
