"""MongoDB connection layer for the invoice processor app."""

from pymongo import MongoClient
from pymongo.errors import PyMongoError

from config import MONGO_DB_NAME, MONGO_URI


client = None
database = None
invoices_collection = None
processing_logs_collection = None


def _initialize_database() -> None:
	global client, database, invoices_collection, processing_logs_collection

	try:
		client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
		database = client[MONGO_DB_NAME]
		invoices_collection = database["invoices"]
		invoices_collection.create_index("extracted_text", name="extracted_text_text_index")
		invoices_collection.create_index("processing_status", name="processing_status_index")
		invoices_collection.create_index("extracted_fields.vendor_name", name="vendor_name_index")
		invoices_collection.create_index("extracted_fields.invoice_date", name="invoice_date_index")
		invoices_collection.create_index("extracted_fields.grand_total", name="grand_total_index")
		invoices_collection.create_index([("created_at", -1)], name="created_at_desc_index")
		processing_logs_collection = database["processing_logs"]
	except PyMongoError as error:
		client = None
		database = None
		invoices_collection = None
		processing_logs_collection = None
		print(f"MongoDB connection error: {error}")


def get_database():
	"""Return the configured MongoDB database object."""

	if database is None:
		raise RuntimeError("MongoDB database is not available")
	return database


_initialize_database()

