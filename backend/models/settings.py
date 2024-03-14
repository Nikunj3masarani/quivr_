import os
from uuid import UUID

from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.embeddings.azure_openai import AzureOpenAIEmbeddings
from logger import get_logger
from models.databases.supabase.supabase import SupabaseDB
from pydantic_settings import BaseSettings, SettingsConfigDict
from supabase.client import Client, create_client
from vectorstore.supabase import SupabaseVectorStore

logger = get_logger(__name__)


class BrainRateLimiting(BaseSettings):
    model_config = SettingsConfigDict(validate_default=False)
    max_brain_per_user: int = 5


class BrainSettings(BaseSettings):
    model_config = SettingsConfigDict(validate_default=False)

    openai_api_key: str = ""
    openai_embeddings_deployment: str = ""
    openai_embeddings_api_version:str  = ""
    openai_embeddings_azure_endpoint: str = ""

    supabase_url: str = ""
    supabase_service_key: str = ""
    resend_api_key: str = "null"
    resend_email_address: str = "brain@mail.quivr.app"
    ollama_api_base_url: str = None
    langfuse_public_key: str = None
    langfuse_secret_key: str = None



class ResendSettings(BaseSettings):
    model_config = SettingsConfigDict(validate_default=False)
    resend_api_key: str = "null"


def get_supabase_client() -> Client:
    settings = BrainSettings()  # pyright: ignore reportPrivateUsage=none
    supabase_client: Client = create_client(
        settings.supabase_url, settings.supabase_service_key
    )
    return supabase_client


def get_supabase_db() -> SupabaseDB:
    supabase_client = get_supabase_client()
    return SupabaseDB(supabase_client)


def get_embeddings():
    settings = BrainSettings()  # pyright: ignore reportPrivateUsage=none
    if settings.ollama_api_base_url:
        embeddings = OllamaEmbeddings(
            base_url=settings.ollama_api_base_url,
        )  # pyright: ignore reportPrivateUsage=none
    else:
        logger.info("*********************************")
        logger.info("openAI API Key {}".format(settings.openai_api_key))
        logger.info("Deployment {}".format(settings.openai_embeddings_deployment))
        logger.info("OpenAI API version {}".format(settings.openai_embeddings_api_version))
        logger.info("Azure Endpoint".format(settings.openai_embeddings_azure_endpoint))
        logger.info("*********************************")

        embeddings = AzureOpenAIEmbeddings(
            openai_api_key=settings.openai_api_key,
            deployment=settings.openai_embeddings_deployment,
            openai_api_version=settings.openai_embeddings_api_version,
            azure_endpoint=settings.openai_embeddings_azure_endpoint
        )

    return embeddings


def get_documents_vector_store() -> SupabaseVectorStore:
    settings = BrainSettings()  # pyright: ignore reportPrivateUsage=none
    embeddings = get_embeddings()
    supabase_client: Client = create_client(
        settings.supabase_url, settings.supabase_service_key
    )
    documents_vector_store = SupabaseVectorStore(
        supabase_client, embeddings, table_name="vectors"
    )
    return documents_vector_store
