\section{Experimental Results}
We evaluate \myalgo{} on synthetic, semi-synthetic, and real-world datasets. We compare it against five representative non-causal and causal RCA approaches based on statistical hypothesis testing (BARO~\citep{baro2024}), $z$-score-based statistical analysis (NSigma~\citep{nsigma2022}), regression-based hypothesis testing (CIRCA~\citep{circa2022}), causal discovery (RCD~\citep{rcd2022}), and intervention-aware causal inference (RCG~\citep{rcg2025}). We adopt their implementation details and hyperparameters from the RCAEval repository~\citep{rcaeval2025}.

\textbf{Metrics:} For evaluating the output rank, we use  
$AC@k \;=\; \frac{1}{|\mathbf{E}|}\sum_{e \in \mathbf{E}}
\frac{\left|\hat{R}_e[1,..,k] \cap R^{*}_{e}\right|}
{\min\!\left(k,\, |R^{*}_{e}|\right)}$, i.e., how many of the true root causes lie in top $k$ and 
$\mathrm{PRR} \;=\; \frac{1}{|\mathbf{E}|}\sum_{e \in \mathbf{E}}
\mathbf{1}\!\left[\,AC@|R^{*}_{e}| = 1\,\right]$, i.e., how many cases all true root causes are at the top (i.e., perfect recovery rate).
We provide more experimental details in Appendix~\ref{appex:exp}.

\textbf{Causal graph:} The causal structural knowledge can be obtained from domain experts or, as recent work shows, through \textit{inexpensive} LLM-based causal discovery~\cite{llmdisc2024a,llmdisc2024b,llmdisc2023}. In real-world setups such as microservice and agentic frameworks, the call graph or system architecture can also be extracted directly. Thus, we do not consider this requirement a major limitation. Nonetheless, \myalgo{} outperforms the baselines in synthetic experiments with both the true graph and a dense graph, and in microservice experiments where no graph is known and we use a sink graph (all variables point to the target node).

\textbf{Sample size:} We evaluate \myalgo across a range of sample sizes, and it performs consistently well: in synthetic experiments ($N=5$k), in the Causal Chamber ($N=1$k), and in microservice setups ($N \sim 700$). More extensive experiments are in Appendix~\ref{sec:sample-sensitivity}.

\textbf{Runtime:} Our proposed method is computationally more expensive than some baselines (e.g., a single root-cause test takes 0.4\,s for RCD vs.\ 41\,s for ours). However, it outperforms them in almost all cases. Thus, our approach targets applications that prioritize reliability under latent confounding over faster runtime.





\subsection{Simulated Datasets}
\label{exp:sim-dat}

We follow~\citep{synthsem2024} to generate synthetic normal
and anomalous data from randomly generated graphs, varying the total
variable count, latent proportion, and edge density. We sample a random DAG
over $p$ observed variables plus $m$ latent confounders,  observed-to-observed edges follow a random ancestral density, and
each latent connects to a small set of observed children. Only observed
variables are written to the dataset---latents are never revealed to the
algorithms. Given exogenous noises, observations are produced by a
nonlinear SEM whose graph and mixing structure are shared across both
regimes.

For each trial in this experiment we draw $5000$ \emph{normal} samples with
baseline noise scales drawn uniformly from $[a_{\min}, a_{\max}]$ and no
interventions, and $5000$ \emph{anomalous} samples in which a subset of
\emph{observed} variables is selected as root causes; for those targets
only, the noise scale is redrawn from a stronger range
$[b_{\min}, b_{\max}]$, while non-targets retain their baseline scales.
Sample sizes vary across experiments and are stated per experiment. Root
causes are therefore noise-scale mechanism shifts, and the intervened
observed nodes are the ground-truth root causes. Intervention strengths can
optionally be made heterogeneous via a per-variable weight schedule.
% which is separate from Assumption~\ref{assump:shift-effectiveness}. 
Methods see
only the observed tables and, where applicable, the graph.

\textbf{Nonlinear Model with Unobserved Confounders:}
With a fixed causal graph of $6$ observed and $4$ unobserved variables, we
vary the confounding strength $\lambda \in \{3, 10\}$, where $\lambda$
(\texttt{latent\_edge\_strength}) uniformly scales all latent-to-observed
coefficients; increasing $\lambda$ makes hidden confounders contribute more
strongly to the observed variables. The causal mechanism of each variable
combines linear mixing with an MLP. In this experiment we take half of the
observed variables as root causes, $\lfloor p/2 \rfloor$ with $p = 6$,
giving $3$ ground-truth root causes. See Section~\ref{appex:nonlin-conf}
for further details.

\textbf{Observation:}
Figure~\ref{fig:latent_str}(a) reports PRR at two levels of latent strength.
\myalgo attains $99\%$ and $84\%$, ahead of every baseline at both settings---BARO
($91\%, 65\%$), NSigma ($86\%, 62\%$), RCG ($85\%, 39\%$), RCD ($74\%, 59\%$),
CIRCA ($62\%, 38\%$)---with every gap significant under a paired test ($p<0.05$).
All methods degrade as confounding strengthens, but \myalgo degrades least
($-15$ points, against $-26$ for BARO and $-46$ for RCG). The reason is structural:
a latent loading on several observed variables shifts all of them together, so a
non-intervened child of a confounder acquires a marginal shift indistinguishable
from a genuinely intervened node, and raising $\lambda$ amplifies exactly that
shared component. \myalgo instead tests whether the \emph{conditional} mechanism
$P(V\mid\mathrm{pa}(V))$ changed, and represents bidirected edges as shared latent
noise, so confounder-induced variation is absorbed by a term the model already
accounts for.


\begin{figure}[t]
    % \centering
     \vspace{-4mm}
    \hspace{-10mm}
        \centering
       \includegraphics[width=1\linewidth]{figures/row_all3_forest_hatch.pdf}
    \caption{Perfect recovery rate (PRR) with $95\%$ confidence intervals.
(a) increasing latent confounding, (b) heterogeneous anomalies where a non-cause can
out-shift a true cause, (c) real causal-chamber data with the root cause observed
vs.\ hidden. \myalgo{} is the only method strong across all three settings; comparisons
use a paired test, so overlapping intervals do not imply the absence of a difference.
    }
    \label{fig:latent_str}
\end{figure}



\textbf{Nonlinear Model with Heterogeneous Anomaly:}
Baselines that performs RCA mainly by measuring marginal-shift assume homogeneous anomaly where the true root-cause nodes exhibit the largest distributional shift among all variables.
This is however, not always realistic with multiple root causes. A shift originating at a root cause $R_1$ propagates to its descendants, and a downstream, non-root-cause descendant can end up with a larger observed shift than another root cause $R_2$ constructing a failure case for these baselines.

We construct such a heterogeneous anomaly by scaling the injected shift magnitude
with topological position, so that shifts accumulate along directed paths and a
downstream descendant out-shifts a genuine upstream cause. In the generated data this
trap fires---some non-root-cause exhibits a larger marginal shift than some true root
cause---in roughly $60\%$ of trials at $n=10$.
We follow the same random ADMG generation process as the previous section, but with no
latent confounders and $\lfloor n/2 \rfloor$ of the observed variables as root causes.
We fix $5000$ normal and $5000$ anomalous samples while varying the number of variables
as $\{5,10\}$.


\textbf{Observation:}
Figure~\ref{fig:latent_str}(b) shows PRR as the system grows. \myalgo achieves the
highest rate at both sizes, $98\%$ at $n=5$ and $54\%$ at $n=10$, significantly better
than all five baselines in both settings ($p<0.05$, paired). The marginal-shift
detectors collapse hardest, to $14\%$ at $n=10$: this is the failure the design targets,
since the construction guarantees that ranking by shift magnitude is systematically
wrong. The graph-based methods do better (RCG $36\%$, CIRCA $34\%$, RCD $28\%$) because
they propagate evidence along edges rather than scoring nodes in isolation, so a
descendant's large shift can be partly explained away by its parents---but none models
the mechanism itself, and their attributions interfere when several causes fire at once.


\textbf{Performance with graph mis-specification:}
The experiments above supply the true ADMG. We next ask what happens when the supplied
graph is wrong. Holding the $\lambda=10$ trials fixed and perturbing \emph{only} the
graph given to \myalgo---so any change is attributable to the graph error alone---we
compare three corruptions against the true-graph control ($88\%$): adding $50\%$
spurious edges of both kinds ($84\%$), deleting $50\%$ of the bidirected edges ($87\%$),
and deleting $50\%$ of both edge types ($76\%$). Neither addition ($p=0.39$) nor
confounder deletion ($p=1.00$) changes accuracy significantly; only deleting directed
edges degrades it ($p=0.019$), and even then \myalgo retains $76\%$. The asymmetry is
expected: if $P$ is Markov to $G$ and $G\subseteq G'$, then $P$ is Markov to $G'$, so a
denser graph makes no false independence claims and over-specification costs parameters
rather than correctness; deleting directed edges instead leaves
$P(V\mid\mathrm{pa}(V))$ fit against an incomplete parent set. Since both deletion arms
remove the \emph{same} confounders, the gap between them isolates directed-edge loss as
the sole failure mode.

\subsection{Real Data: Causal Chamber}
\label{exp:chamber}

We evaluate on the Causal Chamber light-tunnel dataset ($52$ cases) in two conditions:
the true root cause \emph{observed}, and the true root cause \emph{hidden}, so that it
acts as a latent confounder among its former children. Hiding a column is what creates
the confounding---its children remain dependent through it, but it is no longer
available to condition on. In the observed condition $|R^*|=1$; in the hidden condition
the hidden cause has between $1$ and $7$ children, all of which must be recovered.

\textbf{Observation:}
Figure~\ref{fig:latent_str}(c) reports PRR in both conditions. Without confounding
\myalgo does not lead: RCG is perfect ($100\%$) and RCD reaches $90\%$, against
\myalgo's $88\%$. With confounding the ordering reverses---\myalgo attains $87\%$
against $73\%$ for the best baseline---and \myalgo is the only method that barely moves
between conditions ($-2$ points, against $-50$ for RCG and $-21$ for RCD), which is the
expected shape for a method built for latent confounding. The advantage is concentrated
in multi-cause instances: restricted to cases with $|R^*|>1$, \myalgo recovers the full
set in $63\%$ of trials while CIRCA reaches $25\%$, RCG $13\%$, and RCD $0\%$. RCD is
perfect with a single root cause and never succeeds with more than one, because its
elimination procedure yields one surviving candidate and leaves the rest of the ranking
uninformative. \myalgo scores each node independently by whether its own mechanism
changed, so the children of a hidden cause all score high and cluster at the top
together.


\subsection{Microservice datasets}
\label{exp:micro}

\textbf{Exp 5:}
We evaluate \myalgo on SockShop, a cloud computing dataset recorded from a
microservice-based replica of a web application specifically designed for evaluating
root cause analysis methods~\citep{microservice2024}. This repository contains performance
issues corresponding to five types of faults: CPU hog (``cpu''), memory leak (``mem''),
disk I/O stress (``disk''), network delay (``delay''), and packet loss (``loss''). Each
issue type is injected five times into services such as carts, catalogue, orders,
payment, and users, resulting in a total of $125$ anomaly datasets. Each dataset
provides roughly $360$ normal and $360$ anomalous observations over $14$ recorded
services. We split each dataset into normal and anomalous segments based on the
injection time. We assume \emph{no} causal structure is available to \myalgo{}: instead
of a call graph we supply a \emph{sink confounded star}, in which every recorded service
points to the service under test and a single shared latent confounds all of them. The
confounder is not cosmetic. A plain sink graph, with directed edges into the target but
no edges among the remaining services, asserts that those services are mutually
independent---which is false in these systems, where co-located services share load and
infrastructure (we measure a mean absolute pairwise correlation of $0.11$ on SockShop
and $0.18$ on Online Boutique in the normal regime). Adding the shared latent removes
that false claim while still encoding no structural knowledge beyond ``any service may
be responsible, and they may share unobserved common causes''. We note that this puts
\myalgo{} at a disadvantage relative to two of the baselines: CIRCA and RCG are given
the true edge-inverted call graph, while NSigma, BARO and RCD do not use a graph at
all.

\textbf{Exp 6:}
We repeat the evaluation on Online Boutique~\citep{microservice2024}, a second
microservice benchmark with the same five fault types injected five times each, again
giving $125$ anomaly datasets, but over $11$ services and with substantially longer
traces (roughly $2100$ normal and $2100$ anomalous observations per case). The graph
treatment is identical to Exp 5---\myalgo{} receives only the sink confounded star,
while CIRCA and RCG receive the true call graph. Because the two benchmarks differ in topology, service
count, and trace length, together they test whether a method's behaviour transfers
across deployments rather than being tuned to one.

\textbf{Results:}
On SockShop, \myalgo attains the highest top-$1$ accuracy overall at $0.89$, ahead of
NSigma ($0.75$), BARO ($0.74$), CIRCA ($0.65$), RCG ($0.54$), and RCD ($0.38$); every
gap is significant under a paired test ($p<0.001$). It is best or tied-best on four of
the five fault types---CPU ($1.00$), memory ($0.96$), delay ($0.92$) and loss
($0.80$)---and its errors are near-misses rather than failures: top-$3$ accuracy reaches
$0.99$ and PRR $0.98$. On Online Boutique \myalgo again ranks first at $0.78$, against
RCG ($0.71$), NSigma ($0.70$), BARO ($0.62$), CIRCA ($0.52$) and RCD ($0.49$), with
top-$3$ accuracy $0.90$ and PRR $0.94$; the margins over BARO, CIRCA and RCD are
significant, while those over NSigma and RCG are not at this sample size.

Two observations are worth drawing out. First, no baseline is consistently second:
NSigma is the strongest competitor on SockShop but BARO falls to fifth on Online
Boutique, and RCG rises from fifth to second. This is consistent
with~\citet{rcgnote2025}, who suggest that BARO may be optimized for SockShop, and
with our own synthetic and Causal Chamber results, where the ranking among baselines
likewise reorders with the setting. \myalgo is the only method that is first on both.
Second, both benchmarks agree on \emph{which} faults are hard: CPU and memory are
recovered almost perfectly ($1.00$ and $0.96$--$1.00$), whereas packet loss is the
weakest category on both ($0.80$ and $0.44$). Loss perturbs a service only
intermittently, so the induced mechanism shift is small relative to normal traffic
variation---a limitation of the anomaly signal itself rather than of the attribution
method, since every baseline degrades on the same category.

Notably, \myalgo{} attains these results without any causal structure, while the two
strongest graph-based baselines are handed the true call graph. Together with the
synthetic mis-specification results (Section~\ref{exp:sim-dat}), this supports the view
that \myalgo{}'s advantage does not depend on access to an accurate structure.
