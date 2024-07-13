#[cfg(test)]
mod test {
    use crate::wavegraph::GraphModel;
    use crate::wavegraph::WaveModel;
    use petgraph::graph::{DefaultIx, NodeIndex};
    use std::collections::HashMap;
    use std::fs::read_to_string;
    use sucds::bit_vectors::Rank9Sel;

    fn parse_graph(
        filename: &str,
        weighted: bool,
    ) -> (usize, usize, (Vec<(usize, usize)>, Vec<f64>)) {
        let lines_as_vec = read_lines(filename);
        let number_of_nodes = lines_as_vec[0].parse::<usize>().unwrap();
        let number_of_edges = lines_as_vec[1].parse::<usize>().unwrap();
        let temp_edges_vec = &lines_as_vec[2..];
        if weighted {
            (
                number_of_nodes,
                number_of_edges,
                parse_weighted(temp_edges_vec, number_of_edges),
            )
        } else {
            (
                number_of_nodes,
                number_of_edges,
                parse_non_weighted(temp_edges_vec, number_of_edges),
            )
        }
    }

    fn parse_weighted(
        temp_edges_vec: &[String],
        number_of_edges: usize,
    ) -> (Vec<(usize, usize)>, Vec<f64>) {
        let mut edges: Vec<(usize, usize)> = Vec::new();
        let mut weights: Vec<f64> = Vec::new();
        for i in 0..number_of_edges {
            let line_iter = temp_edges_vec[i].split(" ");
            let line = line_iter.collect::<Vec<&str>>();

            edges.push((
                line[0].parse::<usize>().unwrap(),
                line[1].parse::<usize>().unwrap(),
            ));
            weights.push(line[2].parse::<f64>().unwrap());
        }
        (edges, weights)
    }

    fn parse_non_weighted(
        temp_edges_vec: &[String],
        number_of_edges: usize,
    ) -> (Vec<(usize, usize)>, Vec<f64>) {
        let mut edges: Vec<(usize, usize)> = Vec::new();
        let weights = vec![1.0; number_of_edges];
        for i in 0..number_of_edges {
            let line_iter = temp_edges_vec[i].split(" ");
            let line = line_iter.collect::<Vec<&str>>();

            edges.push((
                line[0].parse::<usize>().unwrap(),
                line[1].parse::<usize>().unwrap(),
            ));
        }
        (edges, weights)
    }

    fn read_lines(filename: &str) -> Vec<String> {
        read_to_string(filename)
            .unwrap()
            .lines()
            .map(String::from)
            .collect()
    }

    fn create_directed_benchmark_graph(
        filename: &str,
        has_weighted_edges: bool,
    ) -> GraphModel<usize, f64, f64, petgraph::Directed> {
        match has_weighted_edges {
            true => create_directed_weighted_benchmark_graph(filename),
            false => create_directed_unweighted_benchmark_graph(filename),
        }
    }

    fn create_undirected_benchmark_graph(
        filename: &str,
        has_weighted_edges: bool,
    ) -> GraphModel<usize, f64, f64, petgraph::Undirected> {
        match has_weighted_edges {
            true => create_undirected_weighted_benchmark_graph(filename),
            false => create_undirected_unweighted_benchmark_graph(filename),
        }
    }

    fn create_directed_weighted_benchmark_graph(
        filename: &str,
    ) -> GraphModel<usize, f64, f64, petgraph::prelude::Directed> {
        let mut graph: GraphModel<usize, f64, f64, petgraph::prelude::Directed> =
            GraphModel::new_directed();
        let parsed_file = parse_graph(filename, true);
        // creates Nodes from 0 to number_of_nodes, which is the first line of the
        // passed file
        for i in 0..parsed_file.0 {
            graph.add_node(i, 1.0);
        }
        // creates Edges from 0 to number_of_edges, which is the second line of the
        // passed file
        for i in 0..parsed_file.1 {
            graph.add_edge(
                NodeIndex::new(parsed_file.2 .0[i].0),
                NodeIndex::new(parsed_file.2 .0[i].1),
                i,
                parsed_file.2 .1[i],
            );
        }
        graph.clone()
    }

    fn create_undirected_weighted_benchmark_graph(
        filename: &str,
    ) -> GraphModel<usize, f64, f64, petgraph::prelude::Undirected> {
        let mut graph: GraphModel<usize, f64, f64, petgraph::prelude::Undirected> =
            GraphModel::new_undirected();
        let parsed_file = parse_graph(filename, true);
        // creates Nodes from 0 to number_of_nodes, which is the first line of the
        // passed file
        for i in 0..parsed_file.0 {
            graph.add_node(i, 1.0);
        }
        // creates Edges from 0 to number_of_edges, which is the second line of the passed
        // file
        for i in 0..parsed_file.1 {
            graph.add_edge(
                NodeIndex::new(parsed_file.2 .0[i].0),
                NodeIndex::new(parsed_file.2 .0[i].1),
                i,
                parsed_file.2 .1[i],
            );
        }
        graph.clone()
    }

    // unweighted being all edges weights are set to 1.0
    fn create_directed_unweighted_benchmark_graph(
        filename: &str,
    ) -> GraphModel<usize, f64, f64, petgraph::prelude::Directed> {
        let mut graph: GraphModel<usize, f64, f64, petgraph::prelude::Directed> =
            GraphModel::new_directed();
        let parsed_file = parse_graph(filename, false);
        // creates Nodes from 0 to number_of_nodes, which is the first line fo the passed file
        for i in 0..parsed_file.0 {
            graph.add_node(i, 1.0);
        }
        // creates Edges from 0 to number_of_edges, which is the second line of the passed file
        for i in 0..parsed_file.1 {
            graph.add_edge(
                NodeIndex::new(parsed_file.2 .0[i].0),
                NodeIndex::new(parsed_file.2 .0[i].1),
                i,
                parsed_file.2 .1[i],
            );
        }
        graph.clone()
    }

    // unweighted being all edge weights are set to 1.0
    fn create_undirected_unweighted_benchmark_graph(
        filename: &str,
    ) -> GraphModel<usize, f64, f64, petgraph::prelude::Undirected> {
        let mut graph: GraphModel<usize, f64, f64, petgraph::prelude::Undirected> =
            GraphModel::new_undirected();
        let parsed_file = parse_graph(filename, false);
        // creates Nodes from 0 to number_of_nodes, which is the first line of the passed file
        for i in 0..parsed_file.0 {
            graph.add_node(i, 1.0);
        }
        // creates Edges from 0 to number_of_edges, which is the seconds line of the passed file
        for i in 0..parsed_file.1 {
            graph.add_edge(
                NodeIndex::new(parsed_file.2 .0[i].0),
                NodeIndex::new(parsed_file.2 .0[i].1),
                i,
                parsed_file.2 .1[i],
            );
        }
        graph.clone()
    }

    #[test]
    #[ignore]
    pub fn bench_tinydg_directed_create() {
        let _ = create_directed_unweighted_benchmark_graph("src/bench/input/tinyDG.txt");
        assert!(true);
    }

    #[test]
    #[ignore]
    pub fn bench_tinydg_undirected_create() {
        let _ = create_undirected_unweighted_benchmark_graph("src/bench/input/tinyDG.txt");
        assert!(true);
    }

    #[test]
    #[ignore]
    pub fn bench_tinydg_directed_create_add_node() {
        let mut graph = create_directed_unweighted_benchmark_graph("src/bench/input/tinyDG.txt");
        let _ = graph.add_node(usize::MAX, 1.0);
        assert!(true);
    }

    #[test]
    #[ignore]
    pub fn bench_tinydg_directed_create_add_edge() {
        let mut graph = create_directed_unweighted_benchmark_graph("src/bench/input/tinyDG.txt");
        let v1 = graph.add_node(usize::MAX, 1.0);
        let v2 = graph.add_node(usize::MAX - 1, 1.0);
        graph.add_edge(v1, v2, usize::MAX, 1.0);
        assert!(true);
    }

    #[test]
    #[ignore]
    pub fn bench_tinydg_undirected_create_add_node() {
        let mut graph = create_undirected_unweighted_benchmark_graph("src/bench/input/tinyDG.txt");
        let _ = graph.add_node(usize::MAX, 1.0);
        assert!(true);
    }

    #[test]
    #[ignore]
    pub fn bench_tinydg_undirected_create_add_edge() {
        let mut graph = create_undirected_unweighted_benchmark_graph("src/bench/input/tinyDG.txt");
        let v1 = graph.add_node(usize::MAX, 1.0);
        let v2 = graph.add_node(usize::MAX - 1, 1.0);
        graph.add_edge(v1, v2, usize::MAX, 1.0);
        assert!(true);
    }
}
