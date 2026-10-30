# Role
Sliding-window rate limiter: class RateLimiter(max_requests: int, window_seconds: float, clock: Callable[[], float] = time.monotonic)

# Interface
- method allow_request() -> bool  (True = allowed, False = denied)

# Constraints
- sliding window of window_seconds; max_requests per window
- time driven ONLY through injectable clock

# Acceptance Criteria
1. 5 requests inside window: first 5 allowed, 6th denied
2. after window passes (fake clock advanced), requests allowed again
3. denied request does not consume window slot

# Test Plan
1. test_five_allowed_sixth_denied
2. test_window_expiry_via_fake_clock
3. test_denied_does_not_consume_slot
