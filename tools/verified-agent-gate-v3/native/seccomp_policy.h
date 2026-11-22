#ifndef GATE_SECCOMP_POLICY_H
#define GATE_SECCOMP_POLICY_H
#include <linux/filter.h>
#include <stddef.h>
size_t gate_build_filter(struct sock_filter *out, size_t capacity, int worker);
int gate_install_worker_filter(void);
int gate_worker_policy_version(void);
#endif
