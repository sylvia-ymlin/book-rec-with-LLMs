import pytest
from unittest.mock import patch, MagicMock
from src.core.metadata_store import MetadataStore
import sqlite3
from pathlib import Path

class TestMetadataStore:
    @pytest.fixture
    def store(self):
        # Reset singleton
        MetadataStore._instance = None
        return MetadataStore()

    def test_singleton(self, store):
        store2 = MetadataStore()
        assert store is store2

    def test_get_book_metadata(self, store):
        # Mock connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        
        with patch.object(MetadataStore, 'connection', new_callable=PropertyMock) as mock_conn_prop:
            mock_conn_prop.return_value = mock_conn
            
            # Mock row
            mock_row = {'isbn13': '123', 'title': 'Test'}
            mock_cursor.fetchone.return_value = mock_row
            
            res = store.get_book_metadata('123')
            assert res['title'] == 'Test'
            mock_cursor.execute.assert_called()

    def test_get_image(self, store):
         with patch.object(MetadataStore, 'connection', new_callable=PropertyMock) as mock_conn_prop:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_conn_prop.return_value = mock_conn
            mock_conn.cursor.return_value = mock_cursor
            
            mock_cursor.fetchone.return_value = {'image': 'test.jpg'}
            assert store.get_image('123') == 'test.jpg'

from unittest.mock import PropertyMock
