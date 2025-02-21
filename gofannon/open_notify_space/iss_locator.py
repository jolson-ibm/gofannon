import jsonschema
import jsonschema.exceptions
from ..base import BaseTool
from ..config import FunctionRegistry

import logging
import json

import requests
from jsonschema import validate

logger = logging.getLogger(__name__)

valid_iss_schema = {
    "type": "object",
    "properties": {
        "message": {"type": "string", "enum": ["success"]},
        "timestamp": {"type": "integer"},
        "iss_position": {
            "type": "object",
            "properties": {
                "longitude": {"type": "string"},
                "latitude": {"type": "string"},
            },
            "required": ["longitude", "latitude"],
        },
    },
    "required": ["message", "timestamp", "iss_position"],
}

error_response_string = "Error: The ISS endpoint returned an error. No location for the ISS can be determined"
error_response_json = {
    "message": "failure",
    "error": "Error: The ISS endpoint returned an error. No location for the ISS can be determined",
}


@FunctionRegistry.register
class IssLocator(BaseTool):
    def __init__(self, name="iss_locator", format_json=True):
        super().__init__()
        self.name = name
        self.format_json = format_json

    @property
    def definition(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Returns the latitude / longitude of the current location of the International Space Station (ISS)",
                "parameters": {
                    "type" : "object",
                    "properties" : {},
                    "required": [],
                },
            },
        }

    # Function returns one of two messages, depending on the requested format:
    #
    # With format_json=False, Returns one of two strings:
    #   "According to OpenNotify.org, the International Space Station can be found at (lat, long) (x,y)"
    #   or
    #   "The ISS endpoint returned an error. No location for the ISS can be determined"
    #
    # With format_json=True (default), the following strings:
    #   "{
    #        "message": "success",
    #       "timestamp": 1739999640,
    #       "iss_position": {"longitude": "-11.6885", "latitude": "-50.0654"},
    #   }"
    #   or
    #   "{
    #       "message": "failure",
    #       "error": "Error: The ISS endpoint returned an error. No location for the ISS can be determined",
    #   }"

    def fn(self):
        base_url = "http://api.open-notify.org/iss-now.json"
        logger.debug(f"Fetching ISS pos from OpenNotify.org at {base_url}")
        if self.format_json:
            response = json.dumps(error_response_json)
        else:
            response = error_response_string

        try:
            http_response = requests.get(base_url)
            response_json = http_response.json()
            # Validate the returned schema is valid.
            validate(response_json, valid_iss_schema)
            # Does not seem to be a way to evaluate strings as floats in a range in jsonschema,
            # other than using a REALLY ugly regex.
            lat = float(response_json["iss_position"]["latitude"])
            long = float(response_json["iss_position"]["longitude"])
            if (lat > -90 and lat < 90) and (long > -180 and long < 180):
                if self.format_json:
                    response = json.dumps(response_json)
                else:
                    response = f"According to OpenNotify.org, the International Space Station can be found at (lat, long) ({lat}, {long})"
            else:
                raise ValueError(f"(latitude, longitude) out of range: ({lat},{long})")
        except requests.exceptions.HTTPError as errh:
            logger.debug(f"HTTP exception: GET at {base_url} returns {errh}")
            pass
        except requests.exceptions.ConnectionError as errc:
            logger.debug(f"HTTP connection exception: GET at {base_url} returns {errc}")
            pass
        except requests.exceptions.Timeout as errt:
            logger.debug(f"HTTP timeout exception: GET at {base_url} returns {errt}")
            pass
        except requests.exceptions.RequestException as err:
            logger.debug(f"Requests exception: GET at {base_url} returns {err}")
            pass
        except jsonschema.exceptions.ValidationError as errj:
            logger.debug(
                f"JSON validation failure GET at {base_url} malformed response: {errj}"
            )
            pass
        except ValueError as errv:
            logger.debug(
                f"Value Exception GET at {base_url} malformed response: {errv}"
            )
            pass
        except Exception as erre:
            logger.debug(f"General exception GET at {base_url} Error: {erre}")
            pass

        return response
