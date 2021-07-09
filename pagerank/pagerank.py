import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # dictionary to be returned
    new_dict = {}
    # case where a page is chosen at random from all pages
    num_pages = len(corpus)
    prob_random = (1 - damping_factor) / num_pages
    for key in corpus.keys():
        new_dict[key] = prob_random
    num_linked = len(corpus[page])
    # if the chosen page has no links, choose from all at random
    if num_linked == 0:
        prob_linked = damping_factor / num_pages
        for key in corpus.keys():
            new_dict[key] = prob_linked
    # if the chosen page has linked pages, choose randomly from those
    else:
        prob_linked = damping_factor / num_linked
        for linked_page in corpus[page]:
            new_dict[linked_page] = new_dict[linked_page] + prob_linked
    # sum of probabilities in dictionary should sum to 1
    check_total = 0
    check_total = [check_total + num for num in new_dict.values()]
    print("Sum of probabilities: ", check_total)
    return new_dict


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # create dictionary to be returned
    pageranks = {}
    for page in corpus.keys():
        pageranks[page] = 0
    # choose starting page at random for first iteration
    (starting_page, _) = random.choice(corpus)
    pageranks[starting_page] = pageranks[starting_page] + 1
    # temp_dict holds dictionary that helps determine next move
    temp_dict = transition_model(corpus, starting_page, damping_factor)
    # iterate through the rest of the number of desired samples
    for _ in range(n - 1):
        # based off probabilities, choose next page to move to
        next_page = random.choices(temp_dict.keys(), weights=temp_dict.values(), k=1)[0]
        # add 1 to count of that page
        pageranks[next_page] = pageranks[next_page] + 1
        # reset temp_dict to new transition model
        temp_dict = transition_model(corpus, next_page, damping_factor)
    # convert numbers of counts to proportions
    total = 0
    total = [total + num for num in pageranks.values()]
    pageranks = {page : (count / total) for (page, count) in pageranks}
    # return counts in form of dictionary
    return pageranks


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
