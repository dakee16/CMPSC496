"""
set_password.py - set an account's password from the command line.

    python -m main.set_password dzm6085@psu.edu

There is no self-service password reset in the app, and there deliberately is
no route that sets one: a reset endpoint reachable from the browser is a way to
take over an account, and email delivery does not exist here to prove ownership.
So this is an ADMIN tool, run by someone who already has the database
credentials in .env - which is the same authority that could edit the row by
hand in SQL anyway.

The password is read with getpass: never echoed, never in shell history, and
never a command-line argument (those are visible to `ps` and land in .zsh_history
in plain text). It is hashed with main.auth.hash_password, so the length limits
and the bcrypt cost are the SAME ones registration applies - a password set here
behaves identically to one set at sign-up.
"""
import getpass
import os
import sys

from dotenv import load_dotenv

from .auth import (MAX_PASSWORD_BYTES, MIN_PASSWORD, AuthError, hash_password,
                   normalize_username)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[2].strip())
        return 2

    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), ".env"))
    from supabase import create_client
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

    username = normalize_username(argv[1])
    rows = sb.table("students").select("id, username, role").eq(
        "username", username).limit(1).execute().data
    if not rows:
        # Say so plainly. This is an admin at a terminal, not a stranger at a
        # login form, so there is nothing to be gained by hiding which accounts
        # exist - and everything to be gained by not silently doing nothing.
        print(f"No account for {username!r}.")
        known = sb.table("students").select("username").execute().data or []
        if known:
            print("Accounts that do exist:")
            for r in known:
                print(f"  {r['username']}")
        return 1

    row = rows[0]
    print(f"Setting the password for {row['username']} (role: {row['role']}).")
    pw = getpass.getpass("New password: ")
    if pw != getpass.getpass("Again: "):
        print("They do not match. Nothing was changed.")
        return 1
    try:
        pw_hash = hash_password(pw)          # same limits registration applies
    except AuthError as e:
        print(f"{e} (between {MIN_PASSWORD} characters and "
              f"{MAX_PASSWORD_BYTES} bytes)")
        return 1

    sb.table("students").update({"password_hash": pw_hash}).eq(
        "id", row["id"]).execute()
    print(f"Done. Sign in as {row['username']} with the new password.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
