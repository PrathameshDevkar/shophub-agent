
import pytest
from fastapi.testclient import TestClient

from apps.store_api.main import app, reset_store

client = TestClient(app)

TOKEN_A = "merchant-a-dev-token"
TOKEN_B = "merchant-b-dev-token"
UNSCOPED_TOKEN = "unscoped-dev-token"


@pytest.fixture(autouse=True)
def clean_store():
    reset_store()
    yield
    reset_store()


def auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def test_auth_me_returns_merchant_identity():
    response = client.get("/auth/me", headers=auth(TOKEN_A))

    assert response.status_code == 200
    assert response.json() == {
        "user": "alice",
        "merchant": "merchant_a",
    }


def test_missing_or_invalid_token_is_rejected():
    assert client.get("/orders").status_code == 401
    assert client.get(
        "/orders", headers=auth("not-a-real-token")
    ).status_code == 401


def test_unscoped_token_fails_closed():
    response = client.get(
        "/auth/me", headers=auth(UNSCOPED_TOKEN)
    )
    print(f"inside the test_unscoped_token_fails_closed:response is {response}")

    assert response.status_code == 200
    assert response.json()["merchant"] is None

    for path in (
        "/orders",
        "/orders/lookup?id=1001",
    ):
        result = client.get(path, headers=auth(UNSCOPED_TOKEN))

        assert result.status_code == 403

    for path in ("/orders/refund", "/orders/cancel"):
        result = client.post(
            path,
            headers=auth(UNSCOPED_TOKEN),
            json={"id": 1001},
        )
        print(f"inside the test_unscoped_token_fails_closed:result is {result}")

        assert result.status_code == 403


def test_tenant_isolation_for_list_and_lookup():
    response_a = client.get("/orders", headers=auth(TOKEN_A))
    response_b = client.get("/orders", headers=auth(TOKEN_B))

    assert response_a.status_code == 200
    assert response_b.status_code == 200

    orders_a = response_a.json()["orders"]
    orders_b = response_b.json()["orders"]

    assert orders_a
    assert orders_b
    assert all(o["merchant"] == "merchant_a" for o in orders_a)
    assert all(o["merchant"] == "merchant_b" for o in orders_b)

    # Order 2001 belongs to merchant B.
    foreign_lookup = client.get(
        "/orders/lookup?id=2001",
        headers=auth(TOKEN_A),
    )

    # Legacy API: HTTP 200 does not mean the lookup succeeded.
    assert foreign_lookup.status_code == 200
    assert foreign_lookup.json() == {
        "status": False,
        "message": "Order not found",
    }


def test_foreign_order_cannot_be_refunded_or_cancelled():
    for path in ("/orders/refund", "/orders/cancel"):
        response = client.post(
            path,
            headers=auth(TOKEN_A),
            json={"id": 2001},
        )
        assert response.status_code == 404
        assert response.json()["detail"] == "Order not found"

    assert client.get(
        "/orders/lookup?id=2001",
        headers=auth(TOKEN_B),
    ).json()["status"] is True


def test_unknown_lookup_uses_legacy_envelope():
    response = client.get(
        "/orders/lookup?id=999999",
        headers=auth(TOKEN_A),
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": False,
        "message": "Order not found",
    }


def test_refund_and_cancel_state_transitions():
    refund = client.post(
        "/orders/refund",
        headers=auth(TOKEN_A),
        json={"id": 1001},
    )
    assert refund.status_code == 200
    assert refund.json()["order"]["status"] == "REFUNDED"

    # The same order cannot be refunded twice.
    second_refund = client.post(
        "/orders/refund",
        headers=auth(TOKEN_A),
        json={"id": 1001},
    )
    assert second_refund.status_code == 409

    cancel = client.post(
        "/orders/cancel",
        headers=auth(TOKEN_A),
        json={"id": 1003},
    )
    assert cancel.status_code == 200
    assert cancel.json()["order"]["status"] == "CANCELLED"

    # Shipped orders cannot be cancelled.
    invalid_cancel = client.post(
        "/orders/cancel",
        headers=auth(TOKEN_A),
        json={"id": 1002},
    )
    assert invalid_cancel.status_code == 409


def test_seed_data_contains_injection_notes():
    assert len(client.get(
        "/orders", headers=auth(TOKEN_A)
    ).json()["orders"]) >= 6

    all_notes = [
        order["customer_note"]
        for order in (
            client.get("/orders", headers=auth(TOKEN_A))
            .json()["orders"]
            + client.get("/orders", headers=auth(TOKEN_B))
            .json()["orders"]
        )
    ]

    assert sum("ignore" in note.lower() for note in all_notes) >= 2
