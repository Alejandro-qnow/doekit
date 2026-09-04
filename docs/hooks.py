"""MkDocs hooks for documentation build."""

from urllib.parse import urljoin, urlsplit

from jinja2 import pass_context


@pass_context
def absolute_url(context, value: str) -> str:
    """Return a fully qualified URL for hreflang and alternate links."""
    if value.startswith(("http://", "https://")):
        return value

    page = context.get("page")
    config = context.get("config")
    canonical = None
    if page is not None:
        canonical = page.canonical_url
    if not canonical:
        canonical = config.site_url

    if value.startswith("/"):
        origin = urlsplit(canonical)
        return f"{origin.scheme}://{origin.netloc}{value}"

    return urljoin(canonical, value)


def on_env(env, **kwargs):
    env.filters["absolute_url"] = absolute_url
