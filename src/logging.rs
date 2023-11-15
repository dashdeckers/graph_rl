use {
    anyhow::Result,
    std::{
        fs::File,
        path::Path,
        sync::Arc,
    },
    tracing::Level,
    tracing_subscriber::{
        fmt::{
            layer,
            writer::MakeWriterExt,
        },
        layer::SubscriberExt,
        util::SubscriberInitExt,
    },
};

pub fn setup_logging(
    path: &dyn AsRef<Path>,
    min_level_file: Option<Level>,
    min_level_stdout: Option<Level>,
) -> Result<()> {
    let log_file = Arc::new(File::create(path)?);

    tracing_subscriber::registry()
        // File writer
        .with(
            layer()
                .with_writer(log_file.with_max_level(match min_level_file {
                    Some(level) => level,
                    None => Level::INFO,
                }))
                .with_ansi(false),
        )
        // Stdout writer
        .with(
            layer()
                .with_writer(std::io::stdout.with_max_level(match min_level_stdout {
                    Some(level) => level,
                    None => Level::INFO,
                }))
                .compact()
                .pretty()
                .with_line_number(true)
                .with_thread_ids(false)
                .with_target(false),
        )
        // Create and set Subscriber
        .init();

    Ok(())
}
