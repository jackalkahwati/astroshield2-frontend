import json
import logging
import os
import unittest
from typing import NamedTuple

import yaml
from jsonschema import validate, ValidationError, RefResolver

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SchemaInfo(NamedTuple):
    schema_file: str
    example_files: list[str]


class TestSchemaValidator(unittest.TestCase):
    def setUp(self):
        self.cur_dir = os.path.dirname(__file__)
        self.root_dir = os.path.abspath(f"{self.cur_dir}/../..")
        self.schema_dir = os.path.join(self.root_dir, "schemas")

    def test_data_validator(self):
        schema_info_list = self._parse_schema_folder(self.schema_dir)
        for schema_info in schema_info_list:
            logger.info("Loading: %s", schema_info)

            # Schema file
            with open(schema_info.schema_file, 'r') as schema_file:
                schema = json.load(schema_file)

                # Example files
                for schema_example_file in schema_info.example_files:
                    with open(schema_example_file, 'r') as example_file:
                        example = json.load(example_file)

                    # Validate
                    resolver = RefResolver(base_uri=f"file://{self.schema_dir}/", referrer=schema)
                    try:
                        logger.info("Validating JSON: [%s]; schema: [%s]", schema_example_file, schema_info.schema_file)
                        validate(instance=example, schema=schema, resolver=resolver)
                        logger.info("JSON validation succeeded!")
                    except ValidationError as e:
                        self.fail(f"JSON validation failed: {e}")

    def _parse_schema_folder(self, folder_name: str) -> list[SchemaInfo]:
        schema_info_list = []

        for root, dirs, files in os.walk(folder_name):
            # Check if the current folder is "examples"
            if os.path.basename(root) == "examples":
                # Get the parent folder to find schema files
                parent_folder = os.path.dirname(root)

                # Collect schema files in the parent folder
                schema_files = [
                    os.path.join(parent_folder, f)
                    for f in os.listdir(parent_folder)
                    if f.endswith(".json") and not f.startswith(".")
                ]

                for schema_file in schema_files:
                    schema_version = os.path.splitext(os.path.basename(schema_file))[0]

                    # Find matching example files in the "examples" folder
                    example_files = [
                        os.path.join(root, f)
                        for f in files
                        if f.startswith(schema_version) and f.endswith(".json")
                    ]

                    if example_files:
                        schema_info_list.append(SchemaInfo(
                            schema_file=schema_file,
                            example_files=example_files,
                        ))
                    else:
                        self.fail(f"No examples were found for validation by schema: {schema_file}")

        return schema_info_list


if __name__ == "__main__":
    unittest.main()
