import asyncio
from io import BytesIO
import unittest
from unittest.mock import patch

from starlette.datastructures import UploadFile

import api


class TestUploadEndpoint(unittest.TestCase):
    def test_upload_stores_file_in_production_bucket_before_chunking(self):
        upload = UploadFile(filename="sample.pdf", file=BytesIO(b"pdf contents"))

        with patch.object(api.s3_client, "put_object") as put_object, patch.object(
            api, "chunk_main"
        ) as chunk_main:
            result = asyncio.run(api.upload_and_chunk(upload, 10.0))

        put_object.assert_called_once_with(
            Bucket="enterprise-document-storage-prod-eu-central-1",
            Key="uploads/sample.pdf",
            Body=b"pdf contents",
        )
        chunk_main.assert_called_once()
        self.assertEqual(
            result["s3_location"],
            "s3://enterprise-document-storage-prod-eu-central-1/uploads/sample.pdf",
        )


if __name__ == "__main__":
    unittest.main()