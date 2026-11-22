# Run the verified gate from your browser

GitHub supplies the Ubuntu machine. Your device only needs a browser. The workflow builds the Lean kernel and native sandbox, validates them on the hosted machine, and then runs a bounded agent task with **one available execution function: `run_verified_python`**.

The model sends source files to that function. The trusted controller copies them into a signed snapshot, includes the complete original instruction and exact operator task, authenticates the handoff, and dispatches through the native sandbox. The model is not given a shell, GitHub credentials, a network tool, or an alternative executor.

This is the execution route of the new hosted agent. It does not reconfigure an existing ChatGPT conversation.

## One-time setup

1. Open the [browser setup page](https://dancrab-verified-gate-setup.danxii.chatgpt.site). Its source is [enroll.html](enroll.html). It generates the authority key locally using native browser cryptography; it contains no pre-generated authority key.
2. In [GitHub environment settings](https://github.com/12cymatics/quanqonscious/settings/environments), create an environment named **`verified-gate`**. Restrict its deployment branches to the reviewed branch used for this workflow. Keep the environment and workflow under the operator's control.
3. Generate an authority on the setup page. Save **`GATE_AUTHORITY_KEY_B64`** as an environment **secret** and **`GATE_AUTHORITY_FINGERPRINT`** as an environment **variable**. Retain the public fingerprint separately as your trust record. Incoming jobs never supply the trusted fingerprint.
4. Save a project OpenAI API key as the environment secret **`OPENAI_API_KEY`**. The API project must have available credits and access to your chosen model. The setup page never accepts or transmits an API key.

No private key belongs in the repository, a workflow input, a task, or a chat message. The GitHub environment contains the two secrets; the public fingerprint is a separate variable. The workflow receives them only in the manual run's controller step, after build and native preflight.

## Start a task

Open [Verified Agent Gate in GitHub Actions](https://github.com/12cymatics/quanqonscious/actions/workflows/verified-agent-gate.yml), choose **Run workflow**, select the reviewed branch, enter the complete task, and select a model available to your API project. GitHub requires the workflow definition to be present on the default branch before the manual run button is available.

The repository is public. Treat workflow input and exported results as public material; put credentials only in the environment's secret fields.

The runner accepts bounded Python jobs using the system standard library and the supplied signed source files. Workers cannot open network sockets, launch executables, fork, or start threads. Each job has fixed resource limits. A failed verification, unsafe output, failed worker, unsupported tool call, or exhausted turn limit ends that run. There is no unrestricted fallback.

## Results

The run retains the model's final response, signed execution receipts, task output, and public audit evidence in its result artifact. A signed archive inventory binds the exported files. Private authority and audit keys are excluded by an explicit export allowlist.

Process completion and a model's final response are recorded separately from independent completion of the requested objective. Inspect the actual delivered outputs for the task's acceptance conditions. Cryptographic signatures authenticate bytes and execution records; they do not prove an English claim or a scientific calculation.

Each workflow invocation has a fresh, private controller state and a new task lineage. Parent links and replay rejection apply within that invocation. Restarting a workflow creates new issuance; the exported archive is verification evidence, not resumable secret-bearing state. Cross-run parent continuity and protection against complete historical rollback require a separate durable trusted registry.

## Source and checks

| File | Responsibility |
| --- | --- |
| `bootstrap.sh` | Installs host dependencies, verifies the pinned Lean toolchain archive, compiles the actual kernel and native launcher, and runs native preflight. |
| `run_batch.py` | Loads the independently enrolled signing identity, binds the exact operator task and GitHub run identity, creates private run state, and starts the controller. |
| `agent_runner.py` | Implements the real HTTPS Responses API loop with exactly one custom execution function and rejects other output routes. |
| `export_audit.py` | Exports an explicit public inventory and signs the exported archive manifest. |
| `enroll.html` | Generates a fresh Ed25519 authority inside the operator's browser and guides GitHub enrollment. |
| `../workflow.template.yml` | Source of `.github/workflows/verified-agent-gate.yml`. |

The original [3494-byte instruction](../source/verified_delegation_instruction.txt) remains byte-exact. Its SHA-256 is `b801c62880a6862d716a6107909494e58cf9a6c0c58921eff89296bf293e5fd2`. The original contract and the exact operator task are delivered with every job. The controller chooses every authenticated parent.

The automatic source-check workflow does not receive production secrets or call the model API. Offline transport tests are labeled as such and execute real gate dispatches for the process boundary. A live API run requires the operator's configured API key; a passing offline check is not described as a live API test.

## Provider documentation

- [GitHub-hosted Ubuntu runners](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
- [Run a workflow from the browser](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow)
- [GitHub deployment environments](https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/manage-environments)
- [OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling)
- [API credits for eligible ChatGPT subscribers](https://developers.openai.com/api/docs/guides/chatgpt-api-credits)
