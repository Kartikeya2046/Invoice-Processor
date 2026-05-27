"""MongoDB connection layer for the invoice processor app."""

from pymongo import MongoClient
from pymongo.errors import PyMongoError

from config import MONGO_DB_NAME, MONGO_URI


client = None
database = None
invoices_collection = None


def _initialize_database() -> None:
	global client, database, invoices_collection

	try:
		client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
		database = client[MONGO_DB_NAME]
		invoices_collection = database["invoices"]
		invoices_collection.create_index("extracted_text", name="extracted_text_text_index")
	except PyMongoError as error:
		client = None
		database = None
		invoices_collection = None
		print(f"MongoDB connection error: {error}")


def get_database():
	"""Return the configured MongoDB database object."""

	if database is None:
		raise RuntimeError("MongoDB database is not available")
	return database


_initialize_database()

