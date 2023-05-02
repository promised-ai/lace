use std::collections::HashMap;

use log::debug;
/// The actual implementation of the `Nop` preprocessor. This would usually go
/// in your main `lib.rs` file.
    use mdbook::{preprocess::{Preprocessor, PreprocessorContext}, book::Book, BookItem};
    use pulldown_cmark::{Parser, Event, CodeBlockKind, Tag};
    use regex::Regex;
use anyhow::anyhow;

use serde::Deserialize;

type GammaMap = HashMap<String, lace_stats::rv::dist::Gamma>;

fn check_deserialize<T>(input: &str) -> anyhow::Result<()>
where
    T: for<'a> Deserialize<'a>,
{
    debug!("attempting to deserialize to {}", core::any::type_name::<T>());
    serde_yaml::from_str::<T>(input)?;
    Ok(())
}

macro_rules! check_deserialize_arm {
    ($input:expr, $name:expr, [$($types:ty),*]) => {
        match $name {
            $(stringify!($types) => check_deserialize::<$types>($input),)*
            t =>  Err(anyhow!("unrecognized type to deserialize to: {t}")),
        }
    }
}
    
fn check_deserialize_dyn(input: &str, type_name: &str) -> anyhow::Result<()> {
    check_deserialize_arm!(input, type_name, [lace_codebook::ColMetadata, Vec<lace_codebook::ColMetadata>, lace_stats::rv::dist::Gamma, HashMap<String, lace_stats::rv::dist::Gamma>, GammaMap])
}

/// A Preprocessor for testing YAML code blocks
pub struct YamlTester;

impl YamlTester {
    pub fn new() -> YamlTester {
        YamlTester
    }
}

impl Preprocessor for YamlTester {
    fn name(&self) -> &str {
        "lace-yaml-tester"
    }

    fn run(&self, _ctx: &PreprocessorContext, book: Book) -> anyhow::Result<Book> {
        debug!("Starting the run");
        let re = Regex::new(r"^yaml.*,deserializeTo=([^,]+)").unwrap();
        for book_item in book.iter() {
            // debug!("Examining item {:?}\n", book_item);
            if let BookItem::Chapter(chapter) = book_item {
                debug!("Examining Chapter {}", chapter.name);
                let parser = Parser::new(&chapter.content);
                let mut code_block = Some(String::new());
                for event in parser {
                    // if let Event::Code(content) = event {
                    //     debug!("Found code: {}", content);
                    // }
                    if let Event::Start(Tag::CodeBlock(CodeBlockKind::Fenced(ref code_block_string))) = event {
                        if re.is_match(&code_block_string) {
                            debug!("YAML Block Start, string={}", code_block_string);
                            code_block=Some(String::new());    
                        }
                    } else if let Event::End(Tag::CodeBlock(CodeBlockKind::Fenced(ref code_block_string))) = event {
                        if let Some(captures) = re.captures(&code_block_string) {
                            debug!("Code Block End, string={}", code_block_string);
                            let target_type = captures.get(1).ok_or(anyhow!("No deserialize type found"))?.as_str();
                            debug!("Underlying Type is {}", target_type);
                            let final_block = code_block.take();
                            // debug!("Code block ended up as\n{}", final_block.unwrap_or("<NO STRING FOUND>".to_string()));
                            let final_block = final_block.ok_or(anyhow!("No YAML text found"))?;
                            debug!("Code block ended up as\n{}", final_block);
                            check_deserialize_dyn(&final_block, target_type)?;                            
                        }
                    } else if let Event::Text(ref text) = event {
                        if let Some(existing) = code_block.as_mut() {
                            existing.push_str(text);
                        }
                        
                        ;
                    }
                }
            }
        }

        Ok(book)
    }
}


//     fn run(&self, ctx: &PreprocessorContext, book: Book) -> Result<Book, Error> {
//         // In testing we want to tell the preprocessor to blow up by setting a
//         // particular config value
//         if let Some(nop_cfg) = ctx.config.get_preprocessor(self.name()) {
//             if nop_cfg.contains_key("blow-up") {
//                 anyhow::bail!("Boom!!1!");
//             }
//         }

//         // we *are* a no-op preprocessor after all
//         Ok(book)
//     }

// #[cfg(test)]
// mod test {
//     use super::*;

//     #[test]
//     fn nop_preprocessor_run() {
//         let input_json = r##"[
//             {
//                 "root": "/path/to/book",
//                 "config": {
//                     "book": {
//                         "authors": ["AUTHOR"],
//                         "language": "en",
//                         "multilingual": false,
//                         "src": "src",
//                         "title": "TITLE"
//                     },
//                     "preprocessor": {
//                         "nop": {}
//                     }
//                 },
//                 "renderer": "html",
//                 "mdbook_version": "0.4.21"
//             },
//             {
//                 "sections": [
//                     {
//                         "Chapter": {
//                             "name": "Chapter 1",
//                             "content": "# Chapter 1\n",
//                             "number": [1],
//                             "sub_items": [],
//                             "path": "chapter_1.md",
//                             "source_path": "chapter_1.md",
//                             "parent_names": []
//                         }
//                     }
//                 ],
//                 "__non_exhaustive": null
//             }
//         ]"##;
//         let input_json = input_json.as_bytes();

//         let (ctx, book) = mdbook::preprocess::CmdPreprocessor::parse_input(input_json).unwrap();
//         let expected_book = book.clone();
//         let result = Nop::new().run(&ctx, book);
//         assert!(result.is_ok());

//         // The nop-preprocessor should not have made any changes to the book content.
//         let actual_book = result.unwrap();
//         assert_eq!(actual_book, expected_book);
//     }
// }
