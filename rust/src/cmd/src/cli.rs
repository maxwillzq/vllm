// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! CLI argument definitions for the `vllm-rs` binary.
//!
//! Python vLLM references:
//! - Engine args: <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/engine/arg_utils.py#L657-L1311>
//! - Environment variables: <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/envs.py#L472>

use std::ffi::{OsStr, OsString};
use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};
use educe::Educe;
use uuid::Uuid;
use vllm_engine_core_client::TransportMode;
pub use vllm_frontend_args::*;
use vllm_managed_engine::ManagedEngineConfig;
use vllm_managed_engine::cli::{ManagedEngineArgs, repartition_managed_engine_args};
use vllm_managed_engine::data_parallel::{DataParallelLbArgs, DataParallelTopology};
use vllm_server::{Config, CoordinatorMode, HttpListenerMode, ParserSelection};
use vllm_tracing::LogLevel;

/// Top-level parser for the `vllm-rs` binary.
#[derive(Debug, Parser)]
#[command(
    name = "vllm-rs",
    about = "Rust frontend and managed-engine CLI for vLLM.",
    version = vllm_build_info::VERSION
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

impl Cli {
    /// The level of the Rust frontend's own logs for this command.
    pub fn log_level(&self) -> LogLevel {
        self.runtime()
            .map_or_else(LogLevel::from_env, SharedRuntimeArgs::effective_log_level)
    }

    fn runtime(&self) -> Option<&SharedRuntimeArgs> {
        match &self.command {
            Command::Serve(args) => Some(&args.runtime),
            Command::Frontend(args) => Some(&args.runtime),
            _ => None,
        }
    }

    pub fn parse() -> Self {
        Self::try_parse_from(std::env::args_os()).unwrap_or_else(|error| error.exit())
    }

    pub fn try_parse_from<I, T>(itr: I) -> Result<Self, clap::Error>
    where
        I: IntoIterator<Item = T>,
        T: Into<OsString>,
    {
        let args: Vec<OsString> = itr.into_iter().map(Into::into).collect();
        let repartitioned_args = repartition_managed_engine_args::<Self>(&args, Some("serve"))?;
        <Self as Parser>::try_parse_from(&repartitioned_args).inspect(|cli| {
            if let Command::Serve(serve) = &cli.command
                && serve.debug_cli
            {
                println!(
                    "Original CLI args: {}\n",
                    args.join(OsStr::new(" ")).display()
                );
                println!(
                    "Repartitioned CLI args: {}\n",
                    repartitioned_args.join(OsStr::new(" ")).display()
                );
                println!(
                    "Passthrough Python args: {}",
                    serve.managed_engine.python_args.join(" ")
                );
                std::process::exit(0);
            }
        })
    }
}

/// Supported top-level CLI commands.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Run the Rust OpenAI frontend as a Python-supervised worker.
    Frontend(FrontendArgs),
    /// Launch a managed Python headless engine, then run the Rust OpenAI
    /// frontend.
    Serve(ServeArgs),
    /// Run vLLM benchmarks.
    #[command(subcommand)]
    Bench(BenchCommand),
    /// Run engine-free request rendering and preprocessing.
    Render(RenderArgs),
}

/// Supported benchmark commands.
#[derive(Debug, Subcommand)]
pub enum BenchCommand {
    /// Benchmark online serving throughput.
    Serve(vllm_bench::BenchServeArgs),
}

/// Arguments for running the Rust frontend as a Python-bootstrapped worker.
#[derive(Educe, Clone, Args, PartialEq, Eq)]
#[educe(Debug)]
pub struct FrontendArgs {
    /// Inherited listening socket file descriptor passed by the Python
    /// supervisor.
    #[arg(long)]
    pub listen_fd: i32,
    /// Inherited gRPC listening socket file descriptor passed by the Python
    /// supervisor. When not set, no gRPC server is started.
    #[arg(long)]
    pub grpc_listen_fd: Option<i32>,
    /// Inherited frontend input ROUTER listener file descriptor.
    #[arg(long)]
    pub input_listener_fd: i32,
    /// Inherited frontend output PULL listener file descriptor.
    #[arg(long)]
    pub output_listener_fd: i32,
    /// Optional Python-owned frontend-side DP coordinator socket address for
    /// external coordinator mode in the bootstrapped frontend path, i.e.,
    /// `stats_update_address`.
    #[arg(long)]
    pub coordinator_address: Option<String>,
    /// First data-parallel engine rank expected to register with this
    /// bootstrapped frontend.
    #[arg(long, default_value_t = 0)]
    pub engine_start_index: u32,
    /// Total number of data-parallel engines expected for this frontend.
    #[arg(long, default_value_t = 1)]
    pub engine_count: usize,
    /// Deployment-wide configured data-parallel size. Defaults to engine count.
    #[arg(long)]
    pub data_parallel_size: Option<usize>,

    /// Shared frontend arguments as one JSON object.
    #[arg(long = "args-json", value_parser = parse_runtime_args_json, value_name = "JSON")]
    pub runtime: SharedRuntimeArgs,
}

impl FrontendArgs {
    /// Convert the CLI arguments into the OpenAI server's runtime config.
    pub fn into_config(self) -> Config {
        let data_parallel_size = self.data_parallel_size.unwrap_or(self.engine_count);
        // The Python supervisor launched the engine and opened the listeners.
        let ready_timeout = self.runtime.ready_timeout();
        self.runtime.into_config(ConfigEndpoints {
            transport_mode: TransportMode::Bootstrapped {
                input_listener_fd: self.input_listener_fd,
                output_listener_fd: self.output_listener_fd,
                engine_start_index: self.engine_start_index,
                engine_count: self.engine_count,
                data_parallel_size,
                ready_timeout,
            },
            coordinator_mode: match self.coordinator_address {
                Some(address) => CoordinatorMode::External { address },
                None => CoordinatorMode::None,
            },
            listener_mode: HttpListenerMode::InheritedFd { fd: self.listen_fd },
            grpc_listener_mode: self.grpc_listen_fd.map(|fd| HttpListenerMode::InheritedFd { fd }),
            manages_engine: false,
        })
    }
}

/// Arguments for the managed-engine mode that spawns Python on behalf of the
/// user.
#[derive(Educe, Clone, Args, PartialEq, Eq)]
#[educe(Debug)]
#[command(override_usage = "vllm-rs serve <MODEL> [OPTIONS] [-- <PYTHON_ARGS>...]")]
pub struct ServeArgs {
    /// Only launch the managed Python headless engine and do not start the Rust
    /// frontend.
    #[arg(long)]
    pub headless: bool,
    /// HTTP bind host for the OpenAI-compatible server.
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,
    /// HTTP bind port for the OpenAI-compatible server.
    #[arg(long, default_value_t = 8000)]
    pub port: u16,
    /// Unix domain socket path. If set, host and port arguments are ignored.
    #[arg(long)]
    pub uds: Option<String>,
    /// TCP port for the gRPC Inference and Control services. When not set, no
    /// gRPC server is started.
    #[arg(long)]
    pub grpc_port: Option<u16>,

    /// Flag to print debug information about CLI argument parsing and exit.
    #[educe(Debug(ignore))]
    #[arg(long, hide = true, env = "VLLM_RS_DEBUG_CLI")]
    pub debug_cli: bool,

    /// Shared frontend arguments.
    #[command(flatten)]
    pub runtime: SharedRuntimeArgs,

    /// Managed Python headless-engine arguments.
    #[command(flatten)]
    pub managed_engine: ManagedEngineArgs,

    /// Data parallel load-balancing arguments.
    #[command(flatten)]
    pub data_parallel_lb: DataParallelLbArgs,
}

impl ServeArgs {
    /// Build the OpenAI-server runtime config used after the managed Python
    /// engine starts.
    pub fn to_frontend_config(
        &self,
        handshake_address: String,
        topology: &DataParallelTopology,
    ) -> Config {
        // Prefer IPC sockets for local engine input/output.
        let (local_input_address, local_output_address) =
            topology.frontend_local_only().then(frontend_ipc_addresses).unzip();
        let listener_mode = match &self.uds {
            Some(path) => HttpListenerMode::BindUnix { path: path.clone() },
            None => HttpListenerMode::BindTcp {
                host: self.host.clone(),
                port: self.port,
            },
        };
        // gRPC follows the HTTP TCP host. With a Unix socket it defaults to IPv4
        // loopback rather than all interfaces, so the side-car is never
        // accidentally network-exposed.
        let grpc_listener_mode = self.grpc_port.map(|port| HttpListenerMode::BindTcp {
            host: match self.uds {
                Some(_) => "127.0.0.1".to_string(),
                None => self.host.clone(),
            },
            port,
        });

        let ready_timeout = self.runtime.ready_timeout();
        self.runtime.clone().into_config(ConfigEndpoints {
            transport_mode: TransportMode::HandshakeOwner {
                handshake_address,
                advertised_host: self.managed_engine.handshake_host.clone(),
                engine_start_index: topology.frontend_engine_start_index(),
                engine_count: topology.frontend_engine_count(),
                data_parallel_size: topology.data_parallel_size,
                ready_timeout,
                local_input_address,
                local_output_address,
            },
            coordinator_mode: CoordinatorMode::MaybeInProc,
            listener_mode,
            grpc_listener_mode,
            // `--data-parallel-size-local 0` runs the frontend without a local engine.
            manages_engine: self.managed_engine.data_parallel_size_local != Some(0),
        })
    }

    /// Build the managed Python-engine spawn configuration with the given
    /// handshake port.
    pub fn to_managed_engine_config(
        &self,
        handshake_port: u16,
        topology: &DataParallelTopology,
        client_handshake_address: Option<String>,
    ) -> ManagedEngineConfig {
        let reasoning_parser = effective_engine_reasoning_parser(
            &self.runtime.reasoning_parser,
            &self.runtime.model.id,
        );
        let profiler_config = self.runtime.profiler_config_json();
        let hf_overrides = (!self.runtime.hf_overrides.is_empty()).then(|| {
            serde_json::to_string(&self.runtime.hf_overrides).expect("JSON object serializes")
        });

        let mut managed_engine = self.managed_engine.clone();
        managed_engine.python_args.extend(self.runtime.model.python_args());
        if let Some(logging_config) = self.runtime.logging_config_json() {
            managed_engine
                .python_args
                .extend(["--logging-config".to_string(), logging_config]);
        }
        managed_engine.into_config(
            self.runtime.model.id.clone(),
            self.runtime.max_logprobs,
            profiler_config,
            reasoning_parser.as_deref(),
            self.runtime.language_model_only,
            self.runtime.disable_log_stats,
            self.runtime.shutdown_timeout,
            handshake_port,
            self.runtime.limit_mm_per_prompt_json(),
            hf_overrides,
            self.runtime.structured_outputs_config_json(),
            topology,
            client_handshake_address,
        )
    }
}

fn effective_engine_reasoning_parser(selection: &ParserSelection, model: &str) -> Option<String> {
    selection.resolve_reasoning_name(model).map(str::to_owned)
}

/// Allocate a fresh IPC handshake endpoint for a frontend that serves only its
/// node's engines (hybrid or external DP LB).
pub fn frontend_ipc_handshake_address() -> String {
    let preferred_base_path = std::env::var_os("VLLM_RPC_BASE_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let handshake = preferred_base_path.join(format!("vllm-rs-h-{}", Uuid::new_v4().simple()));
    format!("ipc://{}", handshake.to_string_lossy())
}

/// Allocate fresh IPC endpoints for one managed frontend instance.
fn frontend_ipc_addresses() -> (String, String) {
    let preferred_base_path = std::env::var_os("VLLM_RPC_BASE_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let input_name = format!("vllm-rs-i-{}", Uuid::new_v4().simple());
    let output_name = format!("vllm-rs-o-{}", Uuid::new_v4().simple());

    let input = preferred_base_path.join(input_name);
    let output = preferred_base_path.join(output_name);

    (
        format!("ipc://{}", input.to_string_lossy()),
        format!("ipc://{}", output.to_string_lossy()),
    )
}

#[cfg(test)]
mod tests;
