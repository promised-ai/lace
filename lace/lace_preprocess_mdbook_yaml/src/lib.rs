use std::collections::HashMap;

use anyhow::anyhow;
use log::debug;
use mdbook::{
    book::Book,
    preprocess::{Preprocessor, PreprocessorContext},
    BookItem,
};
use pulldown_cmark::{CodeBlockKind, Event, Parser, Tag};
use regex::Regex;

use serde::Deserialize;

type GammaMap = HashMap<String, lace_stats::rv::dist::Gamma>;

fn check_deserialize_yaml<T>(input: &str) -> anyhow::Result<()>
where
    T: for<'a> Deserialize<'a>,
{
    debug!(
        "attempting to deserialize to {}",
        core::any::type_name::<T>()
    );
    serde_yaml::from_str::<T>(input)?;
    Ok(())
}

fn check_deserialize_json<T>(input: &str) -> anyhow::Result<()>
where
    T: for<'a> Deserialize<'a>,
{
    debug!(
        "attempting to deserialize to {}",
        core::any::type_name::<T>()
    );
    serde_json::from_str::<T>(input)?;
    Ok(())
}

macro_rules! check_deserialize_arm {
    ($input:expr, $name:expr, $format:expr, [$($types:ty),*]) => {
        match $format {
            "yaml" => match $name {
                $(stringify!($types) => check_deserialize_yaml::<$types>($input),)*
                t =>  Err(anyhow!("unrecognized type to deserialize to: {t}")),
            },
            "json" => match $name {
                $(stringify!($types) => check_deserialize_json::<$types>($input),)*
                t =>  Err(anyhow!("unrecognized type to deserialize to: {t}")),
            },
            f => Err(anyhow!("unrecognized serialization format {f}"))
        }
    }
}

fn check_deserialize_dyn(
    input: &str,
    type_name: &str,
    format: &str,
) -> anyhow::Result<()> {
    check_deserialize_arm!(
        input,
        type_name,
        format,
        [
            GammaMap,
            lace_codebook::ColMetadata,
            lace_codebook::ColMetadataList
        ]
    )
}

/// A Preprocessor for testing YAML code blocks
#[derive(Default)]
pub struct YamlTester;

impl YamlTester {
    pub fn new() -> YamlTester {
        YamlTester
    }

    fn examine_chapter_content(
        &self,
        content: &str,
        re: &Regex,
    ) -> anyhow::Result<()> {
        let parser = Parser::new(content);
        let mut code_block = Some(String::new());

        for event in parser {
            match event {
                Event::Start(Tag::CodeBlock(CodeBlockKind::Fenced(
                    ref code_block_string,
                ))) => {
                    if re.is_match(code_block_string) {
                        debug!(
                            "YAML Block Start, identifier string={}",
                            code_block_string
                        );
                        code_block = Some(String::new());
                    }
                }
                Event::End(Tag::CodeBlock(CodeBlockKind::Fenced(
                    ref code_block_string,
                ))) => {
                    if let Some(captures) = re.captures(code_block_string) {
                        debug!(
                            "Code Block End, identifier string={}",
                            code_block_string
                        );

                        let serialization_format = captures
                            .get(1)
                            .ok_or(anyhow!("No serialization format found"))?
                            .as_str();

                        let target_type = captures
                            .get(2)
                            .ok_or(anyhow!("No deserialize type found"))?
                            .as_str();
                        debug!(
                            "Target deserialization type is {}",
                            target_type
                        );

                        let final_block = code_block.take();
                        let final_block =
                            final_block.ok_or(anyhow!("No YAML text found"))?;
                        debug!("Code block ended up as\n{}", final_block);

                        check_deserialize_dyn(
                            &final_block,
                            target_type,
                            serialization_format,
                        )?;
                    }
                }
                Event::Text(ref text) => {
                    if let Some(existing) = code_block.as_mut() {
                        existing.push_str(text);
                    }
                }
                _ => (),
            }
        }

        Ok(())
    }
}

impl Preprocessor for YamlTester {
    fn name(&self) -> &str {
        "lace-yaml-tester"
    }

    fn run(
        &self,
        _ctx: &PreprocessorContext,
        book: Book,
    ) -> anyhow::Result<Book> {
        debug!("Starting the run");
        let re = Regex::new(r"^(yaml|json).*,deserializeTo=([^,]+)").unwrap();
        for book_item in book.iter() {
            if let BookItem::Chapter(chapter) = book_item {
                debug!("Examining Chapter {}", chapter.name);
                self.examine_chapter_content(&chapter.content, &re)?;
            }
        }

        Ok(book)
    }
}

#[cfg(test)]
mod test {
    // use super::*;

    #[test]
    fn dummy() {
        assert!(true);
    }
}
