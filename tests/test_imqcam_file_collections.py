import unittest

from imqcam_file_collections import (
    download_folder_for_results,
    fetch_datafiles,
    safe_folder_name,
)


class MockClient:
    def __init__(self, pages):
        self.pages = pages
        self.requests = []

    def get(self, path, parameters):
        self.requests.append((path, parameters.copy()))
        return self.pages.get(parameters["offset"], [])


class FileCollectionTests(unittest.TestCase):
    def test_filters_matching_items_and_preserves_settings(self):
        client = MockClient(
            {
                0: [
                    {
                        "_id": "matching",
                        "name": "scan.ang",
                        "meta": {
                            "igsn": "CMXMAG00001-001, CMXMAG00001-002",
                            "composition": "IN718",
                        },
                    },
                    {
                        "_id": "wrong-extension",
                        "name": "scan.ctf",
                        "meta": {"igsn": "CMXMAG00001-001", "composition": "IN718"},
                    },
                ]
            }
        )

        results = fetch_datafiles(
            client,
            igsn="cmxmag00001-002",
            composition="in718",
            file_extension="ang",
            page_size=2,
        )

        self.assertEqual(["matching"], [item["_id"] for item in results.items])
        self.assertEqual(("cmxmag00001-002",), results.settings.igsns)
        self.assertEqual(".ang", results.settings.file_extension)

    def test_fetches_all_pages_and_reports_final_progress(self):
        client = MockClient(
            {
                0: [{"_id": "first", "name": "first.ang", "meta": {"igsn": "A"}}],
                1: [{"_id": "second", "name": "second.ang", "meta": {"igsn": "A"}}],
                2: [],
            }
        )
        progress = []

        results = fetch_datafiles(
            client,
            page_size=1,
            progress_callback=lambda page, fetched, matched: progress.append(
                (page, fetched, matched)
            ),
            progress_every=1,
        )

        self.assertEqual(["first", "second"], [item["_id"] for item in results.items])
        self.assertEqual([0, 1, 2], [request[1]["offset"] for request in client.requests])
        self.assertEqual([(1, 1, 1), (2, 2, 2)], progress)

    def test_rejects_repeated_pages(self):
        page = [{"_id": "same", "name": "same.ang", "meta": {"igsn": "A"}}]
        client = MockClient({0: page, 1: page})

        with self.assertRaisesRegex(RuntimeError, "repeated"):
            fetch_datafiles(client, page_size=1)

    def test_download_folder_uses_captured_settings(self):
        client = MockClient(
            {
                0: [
                    {
                        "_id": "item",
                        "name": "scan.ang",
                        "meta": {"igsn": "CMXMAG00001-001", "composition": "IN718"},
                    }
                ]
            }
        )
        results = fetch_datafiles(
            client,
            igsn="CMXMAG00001-001",
            data_type="ebsd_raw",
            composition="IN718",
            file_extension=".ang",
            page_size=2,
        )

        self.assertEqual(
            "downloads/ebsd_raw/CMXMAG00001-001/IN718/.ang",
            download_folder_for_results(".", results.settings).as_posix(),
        )

    def test_rejects_dot_directory_names(self):
        with self.assertRaisesRegex(ValueError, "not allowed"):
            safe_folder_name("..", "fallback")


if __name__ == "__main__":
    unittest.main()
