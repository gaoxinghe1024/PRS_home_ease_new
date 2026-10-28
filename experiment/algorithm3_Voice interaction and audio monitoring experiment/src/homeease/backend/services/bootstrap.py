"""Initial setup: subject/room, accounts, VAPID keys and local HTTPS certificates.

No fixed public passwords: passwords are random unless provided via the
environment, and are printed once to the operator.
"""

from __future__ import annotations

import base64
import datetime as dt
import ipaddress
import os
from pathlib import Path
from typing import Any

from sqlalchemy import select

from homeease.backend.services.security import hash_password, random_password
from homeease.backend.storage.db import Database
from homeease.backend.storage.models import Contact, Room, Subject, User
from homeease.core.clock import utc_now
from homeease.core.config import AppConfig
from homeease.core.paths import secrets_dir, var_dir


def ensure_subject(db: Database, cfg: AppConfig, display_name: str = "Older adult") -> None:
    with db.session() as s:
        if s.get(Subject, cfg.runtime.subject_id) is None:
            s.add(Subject(id=cfg.runtime.subject_id, display_name=display_name, timezone=cfg.runtime.timezone, carriage_setting="unknown", settings={}, created_at=utc_now()))
            s.flush()
        if s.get(Room, cfg.runtime.room_id) is None:
            s.add(Room(id=cfg.runtime.room_id, subject_id=cfg.runtime.subject_id, name=cfg.runtime.room_id.replace("-", " ").title(), camera_device_id="camera-01", zone=list(cfg.acquisition.zone), zone_version=1))


def ensure_user(db: Database, username: str, role: str, display_name: str, password: str | None = None) -> tuple[int, str | None]:
    with db.session() as s:
        u = s.execute(select(User).where(User.username == username)).scalar_one_or_none()
        if u is not None:
            return u.id, None
        pw = password or random_password()
        u = User(username=username, display_name=display_name, role=role, password_hash=hash_password(pw), created_at=utc_now())
        s.add(u)
        s.flush()
        return u.id, pw


def ensure_contact(db: Database, cfg: AppConfig, name: str, user_id: int | None, priority: int, relation: str = "family", channels: list[str] | None = None) -> int:
    with db.session() as s:
        c = s.execute(select(Contact).where(Contact.subject_id == cfg.runtime.subject_id, Contact.user_id == user_id, Contact.name == name)).scalar_one_or_none()
        if c is not None:
            return c.id
        c = Contact(subject_id=cfg.runtime.subject_id, user_id=user_id, name=name, relation=relation, priority=priority, channels=channels or ["inapp"], enabled=True, created_at=utc_now())
        s.add(c)
        s.flush()
        return c.id


def link_elder(db: Database, cfg: AppConfig, user_id: int) -> None:
    with db.session() as s:
        subj = s.get(Subject, cfg.runtime.subject_id)
        if subj is not None:
            subj.elder_user_id = user_id


def init_database(db: Database, cfg: AppConfig, demo: bool = False, admin_password: str | None = None) -> dict[str, Any]:
    db.migrate()
    ensure_subject(db, cfg)
    created: dict[str, Any] = {}
    _, pw = ensure_user(db, "admin", "admin", "Administrator", admin_password or os.environ.get("HOMEEASE_ADMIN_PASSWORD"))
    if pw:
        created["admin"] = pw
    if demo:
        cg1, p1 = ensure_user(db, "caregiver1", "caregiver", "Primary caregiver", os.environ.get("HOMEEASE_DEMO_PASSWORD"))
        cg2, p2 = ensure_user(db, "caregiver2", "caregiver", "Second contact", os.environ.get("HOMEEASE_DEMO_PASSWORD"))
        el, p3 = ensure_user(db, "elder", "elder", "Older adult", os.environ.get("HOMEEASE_DEMO_PASSWORD"))
        ensure_contact(db, cfg, "Primary caregiver", cg1, 1)
        ensure_contact(db, cfg, "Second contact", cg2, 2)
        link_elder(db, cfg, el)
        for k, v in (("caregiver1", p1), ("caregiver2", p2), ("elder", p3)):
            if v:
                created[k] = v
    return created


# ------------------------------------------------------------------ VAPID
def ensure_vapid_keys() -> str:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    d = secrets_dir()
    priv, pub = d / "vapid_private.pem", d / "vapid_public.txt"
    if priv.is_file() and pub.is_file():
        return pub.read_text().strip()
    key = ec.generate_private_key(ec.SECP256R1())
    priv.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    os.chmod(priv, 0o600)
    raw = key.public_key().public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
    b64 = base64.urlsafe_b64encode(raw).decode().rstrip("=")
    pub.write_text(b64)
    return b64


# ---------------------------------------------------------------- certificates
def generate_certs(hosts: list[str], out_dir: Path | None = None, days: int = 365) -> dict[str, str]:
    """Create a local CA and a server certificate signed by it.

    Install ``ca.crt`` (never the key) on the phone to make the LAN HTTPS origin
    trusted; DeviceMotion and Service Workers need a secure context.
    """
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    out = out_dir or (var_dir() / "certs")
    out.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc)
    ca_key_p, ca_crt_p = out / "ca.key", out / "ca.crt"
    if ca_key_p.is_file() and ca_crt_p.is_file():
        ca_key = serialization.load_pem_private_key(ca_key_p.read_bytes(), None)
        ca_crt = x509.load_pem_x509_certificate(ca_crt_p.read_bytes())
    else:
        ca_key = ec.generate_private_key(ec.SECP256R1())
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "HomeEase Local CA"), x509.NameAttribute(NameOID.ORGANIZATION_NAME, "HomeEase (local)")])
        ca_crt = (
            x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(ca_key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - dt.timedelta(minutes=5)).not_valid_after(now + dt.timedelta(days=3650))
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .add_extension(x509.KeyUsage(digital_signature=True, key_cert_sign=True, crl_sign=True, content_commitment=False, key_encipherment=False, data_encipherment=False, key_agreement=False, encipher_only=False, decipher_only=False), critical=True)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
            .sign(ca_key, hashes.SHA256())
        )
        ca_key_p.write_bytes(ca_key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
        os.chmod(ca_key_p, 0o600)
        ca_crt_p.write_bytes(ca_crt.public_bytes(serialization.Encoding.PEM))
    key = ec.generate_private_key(ec.SECP256R1())
    sans: list[x509.GeneralName] = []
    for h in hosts:
        try:
            sans.append(x509.IPAddress(ipaddress.ip_address(h)))
        except ValueError:
            sans.append(x509.DNSName(h))
    crt = (
        x509.CertificateBuilder().subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, hosts[0])])).issuer_name(ca_crt.subject)
        .public_key(key.public_key()).serial_number(x509.random_serial_number()).not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(days=min(days, 397)))
        .add_extension(x509.SubjectAlternativeName(sans), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
        .sign(ca_key, hashes.SHA256())
    )
    (out / "server.key").write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL, serialization.NoEncryption()))
    os.chmod(out / "server.key", 0o600)
    (out / "server.crt").write_bytes(crt.public_bytes(serialization.Encoding.PEM) + ca_crt.public_bytes(serialization.Encoding.PEM))
    return {"ca_cert": str(ca_crt_p), "server_cert": str(out / "server.crt"), "server_key": str(out / "server.key"), "hosts": ",".join(hosts)}


def lan_addresses() -> list[str]:
    import socket

    addrs = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        addrs.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    return sorted(a for a in addrs if not a.startswith("127."))
