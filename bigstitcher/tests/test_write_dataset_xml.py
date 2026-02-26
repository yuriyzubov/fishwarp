"""Test to verify _write_dataset_xml output is preserved after refactoring."""

import tempfile
from pathlib import Path


from bigstitcher.to_bigstitcher import _write_dataset_xml


# Sample test data (ViewSetups are unique per tile×channel — no 'timepoint' key)
SAMPLE_VIEW_SETUPS = [
    {
        'id': 0, 'name': 'tile_0', 'size': (512, 512, 100),
        'voxel_size': (0.5, 0.5, 1.0), 'tile_id': 0,
        'tile_name': 'tile_0', 'channel_id': 0,
    },
    {
        'id': 1, 'name': 'tile_1', 'size': (512, 512, 100),
        'voxel_size': (0.5, 0.5, 1.0), 'tile_id': 1,
        'tile_name': 'tile_1', 'channel_id': 0,
    },
]

SAMPLE_ZGROUPS = [
    {'setup': 0, 'tp': 0, 'path': 'tile_0.zarr', 'indices': '0 0'},
    {'setup': 1, 'tp': 0, 'path': 'tile_1.zarr', 'indices': '0 0'},
]

# Expected XML output.
# Calibration affine encodes voxel_size=(0.5, 0.5, 1.0) on the diagonal.
EXPECTED_XML = """\
<?xml version="1.0" ?>
<SpimData version="0.2">
  <BasePath type="relative">.</BasePath>
  <SequenceDescription>
    <ImageLoader format="bdv.multimg.zarr" version="3.0">
      <zarr type="relative">dataset.zarr</zarr>
      <zgroups>
        <zgroup setup="0" tp="0" path="tile_0.zarr" indicies="0 0"/>
        <zgroup setup="1" tp="0" path="tile_1.zarr" indicies="0 0"/>
      </zgroups>
    </ImageLoader>
    <ViewSetups>
      <ViewSetup>
        <id>0</id>
        <name>tile_0</name>
        <size>512 512 100</size>
        <voxelSize>
          <unit>micrometer</unit>
          <size>0.5 0.5 1.0</size>
        </voxelSize>
        <attributes>
          <illumination>0</illumination>
          <channel>0</channel>
          <tile>0</tile>
          <angle>0</angle>
        </attributes>
      </ViewSetup>
      <ViewSetup>
        <id>1</id>
        <name>tile_1</name>
        <size>512 512 100</size>
        <voxelSize>
          <unit>micrometer</unit>
          <size>0.5 0.5 1.0</size>
        </voxelSize>
        <attributes>
          <illumination>0</illumination>
          <channel>0</channel>
          <tile>1</tile>
          <angle>0</angle>
        </attributes>
      </ViewSetup>
      <Attributes name="illumination">
        <Illumination>
          <id>0</id>
          <name>0</name>
        </Illumination>
      </Attributes>
      <Attributes name="channel">
        <Channel>
          <id>0</id>
          <name>channel_0</name>
        </Channel>
      </Attributes>
      <Attributes name="tile">
        <Tile>
          <id>0</id>
          <name>tile_0</name>
        </Tile>
        <Tile>
          <id>1</id>
          <name>tile_1</name>
        </Tile>
      </Attributes>
      <Attributes name="angle">
        <Angle>
          <id>0</id>
          <name>0</name>
        </Angle>
      </Attributes>
    </ViewSetups>
    <Timepoints type="pattern">
      <integerpattern>0</integerpattern>
    </Timepoints>
    <MissingViews/>
  </SequenceDescription>
  <ViewRegistrations>
    <ViewRegistration timepoint="0" setup="0">
      <ViewTransform type="affine">
        <Name>calibration</Name>
        <affine>0.5 0.0 0.0 0.0 0.0 0.5 0.0 0.0 0.0 0.0 1.0 0.0</affine>
      </ViewTransform>
    </ViewRegistration>
    <ViewRegistration timepoint="0" setup="1">
      <ViewTransform type="affine">
        <Name>calibration</Name>
        <affine>0.5 0.0 0.0 0.0 0.0 0.5 0.0 0.0 0.0 0.0 1.0 0.0</affine>
      </ViewTransform>
    </ViewRegistration>
  </ViewRegistrations>
  <ViewInterestPoints/>
  <BoundingBoxes/>
  <PointSpreadFunctions/>
  <StitchingResults/>
  <IntensityAdjustments/>
</SpimData>"""


def test_write_dataset_xml_output():
    """Test that _write_dataset_xml produces expected output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        xml_path = Path(tmpdir) / 'dataset.xml'
        _write_dataset_xml(
            xml_path, SAMPLE_VIEW_SETUPS, SAMPLE_ZGROUPS,
            'micrometer', (0.5, 0.5, 1.0), ['channel_0']
        )

        with open(xml_path, 'r') as f:
            actual = f.read()

        assert actual == EXPECTED_XML
