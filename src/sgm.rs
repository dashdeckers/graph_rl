use std::hash::Hash;
use std::fmt::{Debug, Display};
use std::collections::HashMap;

use tracing::{instrument, info};
use ordered_float::OrderedFloat;
use petgraph::graph::{Graph, NodeIndex};
use petgraph::dot::Dot;
use candle_core::Tensor;

use crate::replay_buffer::ReplayBuffer;


#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug)]
pub struct State {
    x: OrderedFloat<f32>,
    y: OrderedFloat<f32>,
}
impl From<Tensor> for State {
    fn from(value: Tensor) -> Self {
        let values = value.squeeze(0).unwrap().to_vec1::<f32>().unwrap();
        Self {
            x: OrderedFloat(values[0]),
            y: OrderedFloat(values[0]),
        }
    }
}
impl Display for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}


impl ReplayBuffer {
    // pub fn dot(&self) -> String {
    //     format!("{}", Dot::new(&self.graph)).to_string()
    // }

    #[instrument(skip(self))]
    pub fn construct_sgm<S: Eq + Hash + Clone + Copy + Debug + Display + From<Tensor>>(
        &self,
        d: fn(&S, &S) -> OrderedFloat<f32>,
        max_dist: OrderedFloat<f32>,
        tau: OrderedFloat<f32>,
    ) -> (Graph<S, OrderedFloat<f32>>, HashMap<S, NodeIndex>) {

        // compute the sparse graph
        let mut graph: Graph<S, OrderedFloat<f32>> = Graph::new();
        let mut indices: HashMap<S, NodeIndex> = HashMap::new();

        // iterate over nodes in the dense graph
        for s1 in self.all_states::<S>() {
            // always accept the first node
            if graph.node_count() == 0 {
                let i1 = graph.add_node(s1);
                indices.insert(s1, i1);

            } else {
                // check if new node is TWC consistent
                let is_twc_consistent =
                    graph
                    .node_weights()
                    .all(|s2| Self::TWC(&s1, s2, tau, d, &graph));

                info!(
                    concat!(
                        "\nTWC consistency of {state:#} with respect to ",
                        "the graph so far: {consistent:#}.",
                        "\nThe graph currently has {n_nodes:#} Nodes and ",
                        "{n_edges:#} Edges.",
                    ),
                    state = s1,
                    consistent = is_twc_consistent,
                    n_nodes = graph.node_count(),
                    n_edges = graph.edge_count(),
                );

                if is_twc_consistent {
                    // add node
                    let i1 = graph.add_node(s1);
                    indices.insert(s1, i1);


                    // add edges
                    let mut edges_to_add = Vec::new();
                    for s2 in graph.node_weights() {
                        // no self edges
                        if s1 == *s2 {continue;}

                        let d_out = d(&s1, s2);
                        let d_in = d(s2, &s1);

                        info!(
                            concat!(
                                "\nmax_dist = {dist}, ",
                                "(d_out: {d_out}, d_in: {d_in})",
                            ),
                            dist = max_dist,
                            d_out = d_out,
                            d_in = d_in,
                        );

                        if d_out < max_dist && d_in < max_dist {
                            edges_to_add.push((i1, indices[s2], d(&s1, s2)));
                            edges_to_add.push((indices[s2], i1, d(s2, &s1)));
                        }
                    }
                    for (a, b, weight) in edges_to_add {
                        graph.add_edge(a, b, weight);
                    }
                }
            }
        }

        info!("\nDotviz: {}", Dot::new(&graph));

        (graph, indices)
    }

    /// (insert equation (3) from SGM here
    ///
    /// TWC(s1, s2, tau, d) is defined as C_in >= tau AND C_out >= tau
    /// given the distance function d
    ///
    /// C_out can be seen as a tau-approximate, Q-irrelevant abstraction.
    /// this means that as we vary tau, we vary the sparsification of the graph
    /// in a way that is irrelevant for the Q-function.
    ///
    /// the (goal-conditioned) Q-function is seen as approximately equivalent
    /// to the true distance function, this implies that distance is interpreted
    /// as expected reward.
    ///
    /// Q(state_s1, action_a, goal_g) ~= d(state_s1, state_s2),
    /// where state_s2 is the results of taking action_a in state_s1
    #[allow(non_snake_case)]
    // #[instrument(skip(graph))]
    fn TWC<S: Eq + Hash>(
        s1: &S,
        s2: &S,
        tau: OrderedFloat<f32>,
        d: fn(&S, &S) -> OrderedFloat<f32>,
        graph: &Graph<S, OrderedFloat<f32>>,
    ) -> bool{
        let c_in = Self::c_in(s1, s2, d, graph);
        let c_out = Self::c_out(s1, s2, d, graph);

        info!("\nC_out is {c_out:#} and C_in is {c_in:#}");

        c_out >= tau && c_in >= tau
    }

    fn c_out<S: Eq + Hash>(
        s1: &S,
        s2: &S,
        d: fn(&S, &S) -> OrderedFloat<f32>,
        graph: &Graph<S, OrderedFloat<f32>>,
    ) -> OrderedFloat<f32> {
        graph
            .node_weights()
            .map(|w| OrderedFloat((d(s1, w) - d(s2, w)).abs()))
            .max()
            .expect("Graph cannot be empty because we always accept the first node")
    }

    fn c_in<S: Eq + Hash>(
        s1: &S,
        s2: &S,
        d: fn(&S, &S) -> OrderedFloat<f32>,
        graph: &Graph<S, OrderedFloat<f32>>,
    ) -> OrderedFloat<f32> {
        graph
            .node_weights()
            .map(|w| OrderedFloat((d(w, s1) - d(w, s2)).abs()))
            .max()
            .expect("Graph cannot be empty because we always accept the first node")
    }
}
