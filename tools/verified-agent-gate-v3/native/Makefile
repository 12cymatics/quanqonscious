CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra -Werror -std=c11 -D_FORTIFY_SOURCE=2 -fstack-protector-strong
LDFLAGS ?= -Wl,-z,relro,-z,now

.PHONY: all clean check
all: gate-sandbox libgate_seccomp.so

gate-sandbox: launcher.c seccomp_policy.c seccomp_policy.h
	$(CC) $(CFLAGS) -fPIE -pie launcher.c seccomp_policy.c $(LDFLAGS) -o $@

libgate_seccomp.so: seccomp_policy.c seccomp_policy.h
	$(CC) $(CFLAGS) -fPIC -shared seccomp_policy.c $(LDFLAGS) -o $@

check: all
	python3 test_native.py

clean:
	rm -f gate-sandbox libgate_seccomp.so
