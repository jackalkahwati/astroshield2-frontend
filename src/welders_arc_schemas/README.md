# Welder's Arc Schemas

## Description
The purpose of this repo is to provide a centralized location for schemas used across the Welder's Arc (WA) message bus topics. The folder structure is laid out in similar fashion to the naming convention of the WA topics.

### Topic Naming Convention
Kafka topic name has this limitation:
```
WARNING: Due to limitations in metric names, topics with a period ('.') or underscore ('_') could " +
          "collide. To avoid issues it is best to use either, but not both.
```

So we suggest use `kebab-case` in topic names. E.g. `launch-detection`.

**Why not camelCase?**

Since the topic name will be used as the folder name, we avoid using camelCase to prevent confusion across different OSes:
* Windows: Case-insensitive and does not preserve case.
* Mac: Case-insensitive but preserves case during submission.
To ensure consistency and compatibility, we opt for case-insensitive, non-camelCase naming conventions.

Example Topics:
```
ss0.sensor.heartbeat
ss0.data.weather.lightning
ss2.satellite-state.the-most-amazing-estimate
```

Example Repo Folder/File Layout:
```
schemas/ss0/sensor/heartbeat/0.1.0.json
schemas/ss0/data/weather/lightning/1.0.1.json
schemas/ss2/satellite-state/the-most-amazing-estimate/2.3.0.json
```

Commonly Shared Schema Folder
This folder contains commonly used schemas such as Kafka message header schema and other shared schemas that can be reused via reference.  
```
schemas/common/message-header/0.1.0.json
```

## Python Unit Test Script
To test json example data files against json schema files, Python unit test is provided under `tests/validator`.
The unit test will scan `schemas/` for schema files and their matching JSON examples to run validation.  

To run the validation, use one of the options to set up the environment.

### Option 1: Python Pipenv Installation
If you are using the Pipenv (installed separately) virtual environment, you can install the needed dependencies via the Pipfile:

```
pipenv install
```

### Option 2: Python Pip Installation
For a standard Pip installation, use the following installation:

```
pip install -r requirements.txt
```

### Run All Schema Validation Tests
```
pytest
```

## Create New Topic Schema with Examples
Follow the existing schema and example folder structure. Create a schema file named with its version, and provide matching JSON example files that begin with the same version string.

## Register New Topic Schema
* Go to the file system, right click `message_topic_pub_sub.csv` and open it in Excel or Google Spreadsheet.
* Insert a row for the new topic schema with the following columns:
  * Topic: topic name (please follow the naming convention: using . and -)
  * Implemented: whether or not this topic is implemented.
  * Traceability Parent: whether or not the messages of this topic will have a parent message.
  * Description: detailed description.

## Message Header Descriptions
### `messageId`
- **Type**: `string`
- **Format**: `uuid`
- **Description**: UUID to uniquely identify the message.
  - The payload can have its own unique ID. E.g. for RSO data objects, they will have their unique IDs.
  - For event-type payloads, the message ID can also serve as the event ID.   

### `messageTime`
- **Type**: `string`
- **Format**: `date-time`
- **Description**: ISO 8601 timestamp of when the message occurred.
  - Event-type payloads may include their own timestamp, indicating the time the event occurred. 

### `messageVersion`
- **Type**: `string`
- **Description**: Indicates the message header schema version.

### `subsystem`
- **Type**: `string`
- **Description**: Subsystem that produced the message, e.g., ss0, ss1, etc.

### `dataProvider`
- **Type**: `string`
- **Description**: The provider or component within the subsystem that produced the message, e.g., ss5 component x.
  - This could be the company unique name within the subsystem.

### `dataType`
- **Type**: `string`
- **Description**: Indicates the payload type or topic name, e.g., 'ss5.launch-detection.launch.v1'.

### `dataVersion`
- **Type**: `string`
- **Description**: Indicates the payload schema version.

### `customProperties`
- **Type**: `object`
- **Additional Properties**: Allowed
- **Description**: Additional custom properties serve as metadata for the payload, providing supplementary information specific to the payload's context or requirements.
  - E.g. in a topic designed to transport images, a property like `dataFormatType` can be included to specify the image format, such as JPEG, PNG, or other supported formats.

### `traceability`
- **Type**: `object`
- **Description**: Traceability information linking to parent resources that help produce the message.
  - This is optional and applies only to payloads with associated parent resources - internl (message) or external data source. 
  - E.g. data source sensors in SS0 may not utilize this, as their payload data does not have an associated parent resource.

#### Traceability Items

##### `internal`
- **Type**: `array`
- **Description**: Array of internal traceability information.
- **Items**:
  - `dataType`
    - **Type**: `string`
    - **Description**: Type of the parent message that contributes to the current message data.
      - This property works together with `messageId`. E.g. the parent message's topic name `ss0.data.launch-detection`.
  - `messageId`
    - **Type**: `string`
    - **Description**: ID of the parent message that contributes to the current messag
      - This property works together with `dataType`. E.g. `messageId` of the parent message.

##### `external`
- **Type**: `array`
- **Description**: Array of external traceability information.
- **Items**:
  - `resourceLink`
    - **Type**: `string`
    - **Description**: Link to the parent external resource that contributes to the current message data.
      - This refers to the external link such as UDL data source URL.

  - `parameters`
    - **Type**: `object`
    - **Additional Properties**: Allowed
    - **Description**: Additional parameters associated with the traceability data.
      - This usually works together with `resourceLink` and provides extra details to the link such as query parameters used.
