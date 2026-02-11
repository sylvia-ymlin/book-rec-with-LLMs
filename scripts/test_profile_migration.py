
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.user.profile_store import get_favorites_with_metadata, _load_store, _migrate_favorites

def test_migration():
    print("Testing profile migration...")
    
    # Check raw store content
    store = _load_store()
    print(f"Raw store keys: {list(store.keys())}")
    if "local" in store:
        print(f"Local user favorites type: {type(store['local'].get('favorites'))}")
        print(f"Local user favorites raw: {store['local'].get('favorites')}")
    
    # Test get_favorites_with_metadata which triggers migration
    try:
        favs = get_favorites_with_metadata("local")
        print(f"\nFetched favorites keys: {list(favs.keys())}")
        print(f"First fav meta: {list(favs.values())[0] if favs else 'None'}")
    except Exception as e:
        print(f"Error fetching favorites: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_migration()
